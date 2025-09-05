import argparse
import logging
import os
import time
import threading
import queue

import cv2
import numpy as np
import torch
from collections import deque

import core.logger as Logger
import model as Model
import core.metrics as Metrics


def preprocess_frame(frame, image_size, device, min_max=(-1, 1)):
    """Convert a BGR OpenCV frame to model input tensor.
    Returns a torch.FloatTensor of shape (1,3,H,W) on CPU (device moved later by model).
    """
    # BGR -> RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # resize to model image size
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32) / 255.0
    # HWC -> CHW
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img).float().unsqueeze(0)
    # scale to min_max (default dataset used (-1,1))
    tensor = tensor * (min_max[1] - min_max[0]) + min_max[0]
    return tensor


def letterbox_image(img, target_w, target_h):
    """Resize img to fit into (target_w, target_h) while keeping aspect ratio.
    Pad with black pixels to reach exact target size.
    img expected in BGR order as numpy array.
    """
    h, w = img.shape[:2]
    if w == 0 or h == 0:
        return np.zeros((target_h, target_w, 3), dtype=img.dtype)
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h))
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas = np.zeros((target_h, target_w, 3), dtype=img.dtype)
    canvas[top:top+new_h, left:left+new_w] = resized
    return canvas


def draw_overlay(img, fid, infer_time, display_times):
    """Draw ID, inference time (ms) and FPS onto img (in-place) at top-left.
    display_times is a deque of recent timestamps used to calculate FPS.
    """
    display_times.append(time.time())
    fps = 0.0
    if len(display_times) >= 2 and (display_times[-1] - display_times[0]) > 1e-6:
        fps = len(display_times) / (display_times[-1] - display_times[0])
    txt = f'ID:{fid}  inf:{infer_time*1000:.1f}ms  FPS:{fps:.1f}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(txt, font, scale, thickness)
    pad = 6
    # background rectangle
    cv2.rectangle(img, (5, 5), (5 + text_w + pad, 5 + text_h + pad), (0, 0, 0), -1)
    # text
    cv2.putText(img, txt, (8, 5 + text_h), font, scale, (0, 255, 0), thickness, cv2.LINE_AA)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_32_256_CLIP-UIE.json')
    parser.add_argument('-s', '--source', type=str, default='0',
                        help='video source; integer camera id or path to video file (default=0)')
    parser.add_argument('--no-display', action='store_true', help='do not show GUI window')
    parser.add_argument('--save-video', action='store_true', help='save output to results folder')
    parser.add_argument('--max-frames', type=int, default=-1, help='stop after this many frames (-1 = unlimited)')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_infer', action='store_true')
    parser.add_argument('--n-timesteps', type=int, default=None,
                        help='override val n_timestep for faster sampling (smaller is faster)')
    parser.add_argument('--queue-size', type=int, default=1,
                        help='frame queue size between capture and inference (default 1)')
    parser.add_argument('--max-age', type=float, default=None,
                        help='drop queued frames older than this many seconds (default 1.0)')
    parser.add_argument('--pause-during-infer', action='store_true',
                        help='pause grabbing new frames while inference is running')
    args = parser.parse_args()

    # parse config and prepare opts (re-using existing helper)
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    # logging
    Logger.setup_logger(None, opt['path']['log'], 'infer_video', level=logging.INFO, screen=True)
    logger = logging.getLogger('base')
    logger.info('Options:')
    logger.info(Logger.dict2str(opt))

    # create results dir
    results_dir = opt['path']['results'] if opt['path'] and opt['path'].get('results') else 'results'
    os.makedirs(results_dir, exist_ok=True)

    # model
    diffusion = Model.create_model(opt)
    # allow overriding number of timesteps at inference for speed-quality tradeoff
    schedule_opt = dict(opt['model']['beta_schedule']['val'])
    if args.n_timesteps is not None:
        schedule_opt['n_timestep'] = int(args.n_timesteps)
        logger.info(f'Overriding val n_timestep -> {schedule_opt["n_timestep"]}')
    diffusion.set_new_noise_schedule(schedule_opt, schedule_phase='val')
    logger.info('Model ready for inference')

    # video capture
    # allow numeric camera id
    try:
        source = int(args.source)
    except Exception:
        source = args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f'Could not open video source: {args.source}')
        return

    image_size = opt['model']['diffusion']['image_size'] if opt['model'] and opt['model'].get('diffusion') else 256

    # optional video writer
    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_path = os.path.join(results_dir, 'infer_video_out.avi')
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 25
        # try to get source frame size to create a combined (orig + processed) writer
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if src_w <= 0 or src_h <= 0:
            src_w, src_h = image_size, image_size
        # processed will be resized to src_w x src_h and concatenated horizontally
        writer = cv2.VideoWriter(out_path, fourcc, fps, (src_w * 2, src_h))
        # store source size for runtime
        writer_src_size = (src_w, src_h)
        logger.info(f'Saving output video to: {out_path} (frame size: {src_w*2}x{src_h})')

    frame_count = 0
    t0 = time.time()

    # Shared objects between capture (main) and inference thread
    frame_queue = queue.Queue(maxsize=max(1, int(args.queue_size)))
    latest_result = {'img': None, 'lock': threading.Lock()}
    stop_event = threading.Event()
    worker_busy = threading.Event()
    # keep last timestamps for FPS estimation
    display_times = deque(maxlen=30)

    def inference_worker():
        logger.info('Inference worker started')
        while not stop_event.is_set():
            try:
                frame_id, tensor, orig_bgr, enq_time = frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                # drop if this queued frame is too old
                age = time.time() - enq_time
                if args.max_age is not None and age > float(args.max_age):
                    try:
                        frame_queue.task_done()
                    except Exception:
                        pass
                    continue

                # mark worker busy and measure inference time
                worker_busy.set()
                t0_inf = time.time()
                data = {'SR': tensor}
                diffusion.feed_data(data)
                diffusion.test(continous=False)
                visuals = diffusion.get_current_visuals(sample=True)
                out_img = Metrics.tensor2img(visuals['SAM'])
                if out_img.ndim == 4:
                    out_img = out_img[0]
                out_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
                infer_time = time.time() - t0_inf
                # store latest result as a pair (original, processed, infer_time)
                with latest_result['lock']:
                    latest_result['img'] = (frame_id, orig_bgr, out_bgr, infer_time)
            except Exception as e:
                logger.exception('Exception in inference worker: %s', e)
            finally:
                try:
                    frame_queue.task_done()
                except Exception:
                    pass
                # clear busy flag after finishing this item
                worker_busy.clear()
        logger.info('Inference worker exiting')

    worker = threading.Thread(target=inference_worker, daemon=True)
    worker.start()

    logger.info('Begin video inference loop. Press q to quit, s to save single frame.')
    while True:
        # optionally pause grabbing when inference is running to free resources
        if args.pause_during_infer and worker_busy.is_set():
            # small sleep - still allow GUI events
            time.sleep(0.005)
            # continue to key handling via waitKey below
            ret = True
            # reuse previous frame variable if available
            try:
                frame
            except NameError:
                ret, frame = cap.read()
        else:
            ret, frame = cap.read()
        if not ret:
            logger.info("End of stream or can't fetch frame.")
            break

        frame_count += 1
        if args.max_frames > 0 and frame_count > args.max_frames:
            break

        # preprocess to tensor (on CPU). Model.feed_data will move it to device
        sr_tensor = preprocess_frame(frame, image_size, device=None, min_max=(-1, 1))

        # Try to enqueue latest frame and its original image, drop if queue full (keeps low latency)
        try:
            orig_copy = frame.copy()
            enq_time = time.time()
            frame_queue.put_nowait((frame_count, sr_tensor, orig_copy, enq_time))
        except queue.Full:
            # if queue full, drop oldest queued item to make room (prefer latest frames)
            try:
                _ = frame_queue.get_nowait()
                try:
                    frame_queue.task_done()
                except Exception:
                    pass
            except Exception:
                pass
            try:
                enq_time = time.time()
                frame_queue.put_nowait((frame_count, sr_tensor, orig_copy, enq_time))
            except Exception:
                # give up if still can't enqueue
                pass
        # display latest inference result if available (pair from latest_result to keep sync)
        display_img = None
        orig_frame = None
        with latest_result['lock']:
            if latest_result['img'] is not None:
                fid, orig_frame, display_img, infer_time = latest_result['img']

        if display_img is not None and orig_frame is not None:
            # resize processed output to match original frame size while keeping aspect ratio (letterbox)
            try:
                target_w, target_h = orig_frame.shape[1], orig_frame.shape[0]
                proc_resized = letterbox_image(display_img, target_w, target_h)
            except Exception:
                proc_resized = letterbox_image(display_img, image_size, image_size)

            # combine original and processed side-by-side
            try:
                combined = np.concatenate([orig_frame, proc_resized], axis=1)
            except Exception:
                of = cv2.resize(orig_frame, (proc_resized.shape[1], proc_resized.shape[0]))
                combined = np.concatenate([of, proc_resized], axis=1)

            # overlay text with frame id, infer time and FPS
            combined = draw_overlay(combined, fid, infer_time, display_times)

            if not args.no_display:
                cv2.imshow('SR Output', combined)

            if writer is not None:
                try:
                    out_frame = combined
                    w_needed = writer_src_size[0] * 2
                    h_needed = writer_src_size[1]
                    if out_frame.shape[1] != w_needed or out_frame.shape[0] != h_needed:
                        out_frame = cv2.resize(out_frame, (w_needed, h_needed))
                    writer.write(out_frame)
                except Exception:
                    pass

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s') and display_img is not None and orig_frame is not None:
            save_path = os.path.join(results_dir, f'frame_{fid:06d}_combined.png')
            cv2.imwrite(save_path, combined)
            logger.info(f'Saved frame to {save_path}')

    # shutdown
    stop_event.set()
    worker.join(timeout=2.0)
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
