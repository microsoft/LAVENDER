from copy import deepcopy
import json
import os
import os.path as op
import argparse
import sys
import pickle
import glob
from tqdm import tqdm
from pathlib import Path
pythonpath = os.path.abspath(
    os.path.dirname(os.path.dirname(__file__)))
print(pythonpath)
sys.path.insert(0, pythonpath)
from utils.tsv_file_ops import tsv_reader, tsv_writer


def main(args):
    # === args =========
    data_root = args.data_root
    dataset = args.dataset
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)
    train_video_ids = pickle.load(
        open(f"{data_root}/img_{dataset}.id2lineidx.pkl", "rb")).keys()
    exclude_vids = set()
    all_txt_jsons = {}
    for txt_file in glob.glob(f"{data_root}/txt_{dataset}-*.json"):
        print(txt_file)
        txt_json = json.load(open(txt_file, "r"))
        all_txt_jsons[txt_file] = txt_json
        for split in ["val", "test"]:
            if split in txt_json:
                for item in tqdm(txt_json[split]):
                    exclude_vids.add(item["video"])

    if dataset == "msvd" or dataset == "msrvtt":
        val_test_cap_annotations = {
            "msrvtt": [
                "./_datasets/MSRVTT-v2/val.caption_coco_format.json",
                "./_datasets/MSRVTT-v2/test.caption_coco_format.json"],
            "msvd": ["./_datasets/MSVD/val.caption_coco_format.json",
            "./_datasets/MSVD/test.caption_coco_format.json"]
        }
        for file in val_test_cap_annotations[dataset]:
            if op.exists(file):
                txt_json = json.load(open(file))
                for item in tqdm(txt_json["annotations"]):
                    vid = Path(item["image_id"]).stem
                    exclude_vids.add(vid)

    train_video_ids = set(train_video_ids)
    print(f"Before filtering, {len(train_video_ids)}")
    train_video_ids = train_video_ids.difference(exclude_vids)
    print(f"After filtering, {len(train_video_ids)}")
    for key, value in all_txt_jsons.items():
        new_value = deepcopy(value)
        output_file = key.replace(data_root, output_folder)
        if "train" in value:
            new_value["train"] = []
            for item in tqdm(value["train"]):
                id_ = item["video"]
                if id_ in train_video_ids:
                    new_value["train"].append(item)
            print(f"Before filtering {output_file}: {len(value['train'])})")
            print(f"After filtering {output_file}: {len(new_value['train'])})")
        json.dump(new_value, open(output_file, "w"))

    print("Checking caption train split")
    train_cap_annotations = {
            "msrvtt": "./_datasets/MSRVTT-v2/train.caption_coco_format.json",
            "msvd": "./_datasets/MSVD/train.caption_coco_format.json"
        }
    if op.exists(train_cap_annotations[dataset]):
        txt_json = json.load(open(train_cap_annotations[dataset]))
        cap_train_vids = set()
        for item in tqdm(txt_json["annotations"]):
            vid = Path(item["image_id"]).stem
            cap_train_vids.add(vid)
        print(f"Other train ids: {len(train_video_ids)}")
        print(f"Cap train ids: {len(cap_train_vids)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        type=str, default='lsmdc')
    parser.add_argument("--data_root",
                        type=str, default='./_datasets/')
    parser.add_argument("--output_folder",
                        type=str, default='./_datasets/multi_task/')
    args = parser.parse_args()
    main(args)
