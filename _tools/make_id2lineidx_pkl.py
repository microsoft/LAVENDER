import argparse
import os
import sys
import pickle
from pathlib import Path
import base64
from tqdm import tqdm
pythonpath = os.path.abspath(
    os.path.dirname(os.path.dirname(__file__)))
print(pythonpath)
sys.path.insert(0, pythonpath)
from utils.tsv_file_ops import load_linelist_file
from utils.tsv_file import TSVFile


def decode_frame(b):
    img = base64.b64decode(b)
    return img


def make_id2lineidx_pkl(input_file, output_file):
    corrupt_ids = set()
    # output_folder = f"output/{args.dataset}/frame_tsv"
    data = TSVFile(input_file)
    lineidx_data = load_linelist_file(
        input_file.replace(".tsv", ".lineidx"))
    id2lineidx = {}
    for idx in tqdm(range(len(data))):
        lineidx = lineidx_data[idx]
        d = data[idx]
        video_id = d[0]
        if len(d) > 2:
            first_frame = decode_frame(d[2])
        else:
            first_frame = decode_frame(d[1])

        if first_frame is None:
            print(video_id)
            corrupt_ids.add(video_id)
        video_id = Path(video_id).stem  # .split("/")[-1].replace(".mp4", "")
        id2lineidx[video_id] = lineidx
    resolved_visual_file = f"{output_file}"
    print("generating visual file for", resolved_visual_file)
    pickle.dump(id2lineidx, open(resolved_visual_file, "wb"))


def main(args):
    make_id2lineidx_pkl(args.input_file, args.output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", help="",
                        type=str, default="")
    parser.add_argument("--output_file", help="",
                        type=str, default="")
    args = parser.parse_args()
    main(args)
