import sys
import os

ASPANFORMER_DIR = 'ml-aspanformer-main'
sys.path.insert(0, ASPANFORMER_DIR)

import demo.demo_utils as demo_utils


def draw_match(img0, img1, corr0, corr1):
    return demo_utils.draw_match(img0, img1, corr0, corr1)


def img_path_to_file_name(img_path):
    # '/path/to/data/Autumn_mini_query/1418134275733607.jpg' --> '1418134275733607'
    return os.path.basename(img_path).split('.')[0]
