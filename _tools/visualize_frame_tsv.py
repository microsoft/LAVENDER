import os, sys
import cv2
import argparse
pythonpath = os.path.abspath(
    os.path.dirname(os.path.dirname(__file__)))
print(pythonpath)
sys.path.insert(0, pythonpath)
from utils.tsv_file_ops import img_from_base64


def main(args):
    with open(args.input_file, "r") as f:
        for line_num in [0, 11429530, 23741208]:
            f.seek(line_num)
            line = f.readline()
            item = line.split("\t")
            key = item[0]
            frames = item[2:]
            for idx, img in enumerate(frames):
                img = img_from_base64(img)
                cv2.imwrite(f"./_prepro/debug/{key}_{idx}.jpg", img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", help="",
                        type=str, default="")
    args = parser.parse_args()
    main(args)
