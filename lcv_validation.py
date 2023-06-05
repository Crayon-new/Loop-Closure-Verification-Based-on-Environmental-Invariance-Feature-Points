import os
import argparse
import cv2
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import draw_match, img_path_to_file_name


def loop_verification(version_str, loop_verification_func, data_root_dir,
                      save_dir='./output', draw_match_results=False):
    os.makedirs(save_dir, exist_ok=True)

    # evaluate
    val_files = [
        "robotcar_qAutumn_dbNight_easy_final.txt",
        "robotcar_qAutumn_dbNight_diff_final.txt",
        "robotcar_qAutumn_dbSunCloud_easy_final.txt",
        "robotcar_qAutumn_dbSunCloud_diff_final.txt",
    ]

    for val_file in val_files:
        scene = val_file.strip(".txt")
        result_dir = os.path.join(save_dir, version_str)
        os.makedirs(result_dir, exist_ok=True)

        print("-------- Processing {} ----------".format(scene))
        print("Loading Data...")

        img_pairs = []
        labels = []
        gt_txt = os.path.join(data_root_dir, val_file)
        fp = open(gt_txt, "r")
        for line in fp:
            line_str = line.split(", ")
            query, reference, gt = line_str[0].strip(), line_str[1].strip(), int(line_str[2].strip())
            query_file = os.path.join(data_root_dir, query)
            reference_file = os.path.join(data_root_dir, reference)
            assert os.path.exists(query_file), f"{query_file} does not exist!"
            assert os.path.exists(reference_file), f"{reference_file} does not exist!"
            img_pairs.append([query_file, reference_file])
            labels.append(gt)
        fp.close()
        labels = np.array(labels)

        print("Loop Verification...")

        match_points_cnt_list = []
        corr_pairs = []
        for query, reference in tqdm(img_pairs):
            corr0, corr1 = loop_verification_func(query, reference)
            match_points_cnt_list.append(len(corr0))
            corr_pairs.append([corr0, corr1])

        if draw_match_results:
            print("Drawing Match Results...")
            match_results_dir = os.path.join(result_dir, "match_results")
            os.makedirs(match_results_dir, exist_ok=True)
            for i, (query, reference) in tqdm(enumerate(img_pairs)):
                corr0, corr1 = corr_pairs[i]
                label = labels[i]

                match_cnt = len(corr0)
                img0, img1 = cv2.imread(query), cv2.imread(reference)
                img_match = draw_match(img0, img1, corr0, corr1)
                img_match_name = f'{label}_{match_cnt}_{img_path_to_file_name(query)}_{img_path_to_file_name(reference)}.png'
                img_match_file = os.path.join(match_results_dir, img_match_name)
                cv2.imwrite(img_match_file, img_match)

        match_points_cnt_file = os.path.join(result_dir, f'match_points_cnt_{scene}.txt')
        np.savetxt(match_points_cnt_file, match_points_cnt_list, fmt='%d')

        # precision-recall curve
        match_points_cnt_arr = np.array(match_points_cnt_list)
        scaled_scores = match_points_cnt_arr / max(match_points_cnt_arr)
        precision, recall, _ = precision_recall_curve(labels, scaled_scores)
        average_precision = average_precision_score(labels, scaled_scores)

        # max recall on 100% precision
        max_recall = max(recall[precision == 1])
        precision_file = os.path.join(result_dir, f'MaxRecall@100P_{scene}.txt')
        np.savetxt(precision_file, [max_recall], fmt='%f')

        pr_curve_fig_file = os.path.join(result_dir, f'pr_curve_{scene}.png')
        label_str = f'{scene} AP={average_precision:.2f} MR@100P={max_recall:.2f}'
        plt.plot(recall, precision, label=label_str)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.title("Precision-Recall Curves for " + version_str)
        plt.savefig(pr_curve_fig_file)
        plt.close()


def get_loop_validators():
    from validators_traditional import loop_verification_SIFT
    from validators_traditional import loop_verification_ORB
    from validators_aspanformer import loop_verification_aspanformer
    from validators_aspanformer import loop_verification_aspanformer_remove_ego
    from validators_aspanformer import loop_verification_aspanformer_remove_ego_bgmask
    loop_validators = {
        'SIFT': loop_verification_SIFT,
        'ORB': loop_verification_ORB,
        'Aspanformer': loop_verification_aspanformer,
        'Aspanformer+RemoveEgo': loop_verification_aspanformer_remove_ego,
        'Aspanformer+RemoveEgo+BgMask': loop_verification_aspanformer_remove_ego_bgmask
    }
    return loop_validators


def main(args):
    data_root_dir = args.data_root_dir
    save_dir = args.save_dir
    draw_match_results = args.draw_match_results

    loop_validators_dict = get_loop_validators()

    for version, loop_verification_func in loop_validators_dict.items():
        version_str = version.replace('+', '_')
        print("-------- Processing {} ----------".format(version_str))
        loop_verification(version_str, loop_verification_func, data_root_dir=data_root_dir,
                          save_dir=save_dir, draw_match_results=draw_match_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_dir', type=str, default='./data',
                        help='root directory of validation dataset')
    parser.add_argument('--save_dir', type=str, default='./output',
                        help='directory to save the results')
    parser.add_argument('--draw_match_results', type=bool, default=False,
                        help='whether to draw the match results')
    args = parser.parse_args()
    main(args)
