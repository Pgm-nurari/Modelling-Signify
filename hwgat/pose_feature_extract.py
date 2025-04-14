import csv
import os
import gc
import pickle
import argparse
import warnings
import importlib
import numpy as np
from decord import VideoReader, cpu
from tqdm import tqdm
from multiprocessing import Pool, Lock, cpu_count, Manager
from pose_modules.keypoint_extract_models import keypoint_model_dict

warnings.filterwarnings('ignore')
num_workers = 4  # Using half of the 8-core CPU
lock = Lock()  # For safe CSV writing

def get_model(name: str):
    if name not in keypoint_model_dict:
        raise ValueError(f"Model '{name}' is not found in keypoint_model_dict!")
    module = importlib.import_module('pose_modules.' + keypoint_model_dict[name]['module'])
    return getattr(module, 'Model')()

def init(root: str, meta: str) -> list:
    vid_names = []
    with open(meta, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            vid_names.append([os.path.split(row[1])[0], row[2], root, row[0]])
    return vid_names

def safe_log_failure(csv_path, vid_id, path, error):
    with lock:
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([vid_id, path, str(error)])

def process_video(args_tuple):
    vid_name, model, out_path, failed_csv_path = args_tuple
    vid_id = vid_name[3]
    output_file = os.path.join(out_path, vid_id + ".pkl")

    if os.path.exists(output_file):
        return None  # Already processed

    try:
        pose_model = get_model(model)
    except Exception as e:
        safe_log_failure(failed_csv_path, vid_id, os.path.join(vid_name[2], vid_name[0], vid_name[1]), f"Model load error: {e}")
        return None

    kp_shape = keypoint_model_dict[model]['shape']
    vid_path = os.path.join(vid_name[2], vid_name[0], vid_name[1])

    if not os.path.exists(vid_path):
        safe_log_failure(failed_csv_path, vid_id, vid_path, "File not found")
        return None

    try:
        cap = VideoReader(vid_path, ctx=cpu(0), num_threads=1)
        num_frames = len(cap)
        vid_height, vid_width = cap[0].shape[:2]
        features = np.zeros((num_frames, *kp_shape))

        for i, image in enumerate(cap):
            try:
                features[i] = pose_model(image.asnumpy())[0]
            except Exception:
                continue  # Skip bad frame

    except Exception as e:
        safe_log_failure(failed_csv_path, vid_id, vid_path, f"Frame extraction error: {e}")
        return None

    storing_data = {
        'feat': features,
        'num_frames': num_frames,
        'vid_loc': vid_name[0],
        'vid_name': vid_name[1],
        'vid_width': vid_width,
        'vid_height': vid_height
    }

    try:
        with open(output_file, "wb") as outfile:
            pickle.dump(storing_data, outfile)
    except Exception as e:
        safe_log_failure(failed_csv_path, vid_id, output_file, f"Pickle write error: {e}")
        return None

    gc.collect()
    return None  # Success

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--meta', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('-m', '--model', type=str, default='mediapipe')
    parser.add_argument('--num_processes', '-p', type=int, default=num_workers)
    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parser()

    root = args.root
    meta = args.meta
    out_path = os.path.join(root, args.out_path)
    model = args.model
    num_processes = args.num_processes

    os.makedirs(out_path, exist_ok=True)
    failed_csv_path = os.path.join(out_path, "failed_videos_log.csv")

    # Initialize the log file
    with open(failed_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Video_ID", "Path", "Error"])

    vid_names = init(root, meta)
    print(f"📌 Total Videos: {len(vid_names)}")

    new_vid_names = [row for row in vid_names if str(row[3]) + ".pkl" not in os.listdir(out_path)]
    print(f"📌 Videos to Process: {len(new_vid_names)}")

    # Prepare tasks
    tasks = [(vid, model, out_path, failed_csv_path) for vid in new_vid_names]

    with Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap_unordered(process_video, tasks), total=len(tasks), desc="Processing Videos"))

    print(f"✅ Processing complete. Any failed videos were logged at: {failed_csv_path}")



