import numpy as np
from pathlib import Path
import os, pickle, argparse, csv
import logging, traceback
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from configs import dataCFG
from constants import *
from pose_modules.keypoint_extract_models import *

# Setup logging
log_filename = f'preprocessing_errors.log'
logging.basicConfig(filename=log_filename, level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Global vars for process sharing
class_map_local = {}
root_local = ''
data_root_path_local = ''
cfg_local = None
feature = ''

def arg_parser():
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--root', type=str, required=True,
                        help='Enter the root directory of the dataset')
    parser.add_argument('-ds', type=str, required=True,
                        help='Enter the dataset\'s name')
    parser.add_argument('--meta', type=str, required=True,
                        help='Enter the dataset\'s metadata.csv')
    parser.add_argument('-dr', '--dataroot', type=str, required=False, default='',
                        help='Enter data root relative path')
    parser.add_argument('-ft', '--feature', type=str, required=False, default=feature_type_list[1],
                        help=f'Enter wheather {feature_type_list} data')
    parser.add_argument('-kpm', '--kp_model', type=str, required=False, default='mediapipe', help=f'Choose from {keypoint_model_dict.keys()}')
    # Parse the argument
    args = parser.parse_args()

    return args



def process_row(row, class_map_local, root_local, data_root_path_local, cfg_local, feature):
    try:
        vid_name = row[0]
        word = row[3].strip()
        split = row[4].strip()

        if word not in class_map_local:
            return None

        mapped_class = class_map_local[word]
        # print(mapped_class)

        result = {
            "vid_name": vid_name,
            "class": mapped_class,
            "split": split
        }

        print(result)

        if feature == 'keypoints':
            data_path = os.path.join(data_root_path_local, vid_name + '.pkl')
            if not os.path.exists(data_path):
                logging.error(f"[Missing File] {data_path} not found.")
                return None

            try:
                data = pickle.load(open(data_path, "rb"))
            except Exception as e:
                logging.error(f"[Pickle Load Error] {vid_name}:\n{traceback.format_exc()}")
                return None

            feat = data.get('feat', data.get(feature))
            if feat is None or 1 in feat.shape or 0 in feat.shape or feat.sum() == 0:
                logging.error(f"[Bad Feature] {vid_name} has empty or invalid features.")
                return None

            try:
                data = cfg_local.data_transform(data)
            except Exception as e:
                logging.error(f"[Transform Error] {vid_name}:\n{traceback.format_exc()}")
                return None

            result['data'] = data
        else:
            result['data_path'] = os.path.join(root_local, row[1])

        return result

    except Exception as e:
        logging.error(f"[Row Error] {row}:\n{traceback.format_exc()}")
        return None


if __name__ == "__main__":
    try:
        args = arg_parser()
        root = args.root
        dataset_name = args.ds
        meta = args.meta
        data_root = args.dataroot
        feature = args.feature
        kp_model = args.kp_model

        cfg = dataCFG(dataset_name, feature, kp_model)
        root_local = root
        cfg_local = cfg
        data_root_path_local = data_root

        if feature == 'keypoints' and data_root == '':
            print("Specify dataroot for keypoints")
            exit(0)

        os.makedirs(f'./input/{dataset_name}', exist_ok=True)
        os.makedirs(f'./output/{dataset_name}', exist_ok=True)

        vid_splits = {'train': [], 'val': [], 'test': []}
        vid_class = {}
        class_map = {}
        data_info = {}

        # Build class map
        with open(meta, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            for row in reader:
                word = row[3].strip()
                if word not in class_map:
                    class_map[word] = len(class_map)
        class_map_local = class_map

        # print(class_map_local)

        # Load rows
        with open(meta, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            rows = list(reader)

        print(f"Processing {len(rows)} rows using 4 CPU cores...")

        results = []

        for row in tqdm(rows, desc="Processing rows"):
            res = process_row(row, class_map, root, data_root, cfg, feature)
            results.append(res)

        for res in results:
            if res is None:
                continue
            vid_name = res["vid_name"]
            vid_class[vid_name] = res["class"]
            if feature == 'keypoints':
                data_info[vid_name] = res["data"]
            else:
                data_info[vid_name] = res["data_path"]
            vid_splits[res["split"]].append(vid_name)
            # print(vid_splits)

        print(f'Unique Words: {len(class_map)}')

        # Save outputs
        with open(cfg.vid_split_path, 'wb') as f:
            pickle.dump(vid_splits, f)
            print("Rewriting output")
        with open(cfg.vid_class_path, 'wb') as f:
            pickle.dump(vid_class, f)

        with Path(cfg.class_map_path).open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['class', 'word'])
            for word, cid in class_map.items():
                writer.writerow([cid, word])

        with open(cfg.data_map_path, 'wb') as f:
            pickle.dump(data_info, f)

    except Exception as e:
        logging.critical(f"[Fatal Error] during full preprocessing:\n{traceback.format_exc()}")
        print("Something went wrong! Check preprocessing_errors.log")
