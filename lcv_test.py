import numpy as np
import cv2
from tqdm import tqdm
import os
import argparse

from utils import draw_match, img_path_to_file_name


def get_loop_validator():
    from validators_aspanformer import loop_verification_aspanformer_remove_ego_bgmask
    return loop_verification_aspanformer_remove_ego_bgmask


def evaluate(loop_verification_func, test_file, data_root_dir, score_threshold,
             save_dir='./output', draw_match_results=False):
    scene = os.path.basename(test_file).strip(".txt")

    print("-------- Processing {} ----------".format(scene))
    print("Loading Data...")

    img_pairs = []

    fp = open(test_file, "r")
    for line in fp:
        line_str = line.split(", ")
        query, reference = line_str[0].strip(), line_str[1].strip()
        query_file = os.path.join(data_root_dir, query)
        reference_file = os.path.join(data_root_dir, reference)
        assert os.path.exists(query_file), f"{query_file} does not exist!"
        assert os.path.exists(reference_file), f"{reference_file} does not exist!"
        img_pairs.append([query_file, reference_file])
    fp.close()

    print("Loop Verification...")

    match_points_cnt_list = []
    corr_pairs = []
    for query, reference in tqdm(img_pairs):
        corr0, corr1 = loop_verification_func(query, reference)
        match_points_cnt_list.append(len(corr0))
        corr_pairs.append([corr0, corr1])

    match_points_cnt_file = os.path.join(save_dir, f'match_points_cnt_{scene}.txt')
    np.savetxt(match_points_cnt_file, match_points_cnt_list, fmt='%d')

    scores = np.array(match_points_cnt_list) / np.max(match_points_cnt_list)
    predict = scores > score_threshold
    predict = predict.astype(int)
    predict_result_file = os.path.join(save_dir, f'predict_result_{scene}.txt')
    np.savetxt(predict_result_file, predict, fmt='%d')

    if draw_match_results:
        print("Drawing Match Results...")
        match_results_dir = os.path.join(save_dir, "match_results")
        os.makedirs(match_results_dir, exist_ok=True)
        for i, (query, reference) in tqdm(enumerate(img_pairs)):
            corr0, corr1 = corr_pairs[i]
            score = scores[i]
            match_cnt = len(corr0)
            img0, img1 = cv2.imread(query), cv2.imread(reference)
            img_match = draw_match(img0, img1, corr0, corr1)
            img_match_name = f'{score}_{match_cnt}_{img_path_to_file_name(query)}_{img_path_to_file_name(reference)}.png'
            img_match_file = os.path.join(match_results_dir, img_match_name)
            cv2.imwrite(img_match_file, img_match)


def main(args):
    score_threshold = args.score_threshold
    test_file = args.test_file
    data_root_dir = args.data_root_dir
    save_dir = args.save_dir
    draw_match_results = args.draw_match_results

    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), f'output_thr_{score_threshold}')
    validator = get_loop_validator()
    evaluate(validator, test_file, data_root_dir, score_threshold, save_dir, draw_match_results)


if __name__ == '__main__':
    # test_files = [
    #     "robotcar_qAutumn_dbNight_val_final.txt",
    #     "robotcar_qAutumn_dbSunCloud_val_final.txt",
    # ]

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, required=True,
                        help='test dataset file, each line is a pair of images, format: query_file, reference_file')
    parser.add_argument('--data_root_dir', type=str, default='./data',
                        help='root directory of test dataset')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='directory to save the results')
    parser.add_argument('--score_threshold', type=float, default=0.15,
                        help='threshold to determine whether a loop is detected, 0~1, default: 0.15')
    parser.add_argument('--draw_match_results', type=bool, default=False,
                        help='whether to draw the match results')
    args = parser.parse_args()

    main(args)
