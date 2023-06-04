import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, auc
import os

data_root_dir = './data'

save_dir = './output'

all_version_str_arr = ['SIFT', 'ORB', 'Aspanformer', 'Aspanformer_RemoveEgo',
                       'Aspanformer_RemoveEgo_BgMask']

val_file_names = [
    "robotcar_qAutumn_dbNight_easy_final.txt",
    "robotcar_qAutumn_dbNight_diff_final.txt",
    "robotcar_qAutumn_dbSunCloud_easy_final.txt",
    "robotcar_qAutumn_dbSunCloud_diff_final.txt",
]
val_files = [os.path.join(data_root_dir, file) for file in val_file_names]
scenes = [file[:-4] for file in val_file_names]


def get_version_result(version_str):
    result_dir = os.path.join(save_dir, version_str)
    if not os.path.exists(result_dir):
        print(f'{result_dir} not exists')
        return None
    result = {}
    for val_file, scene in zip(val_files, scenes):
        predict_file = os.path.join(result_dir, f'match_points_cnt_{scene}.txt')
        if not os.path.exists(predict_file):
            print(f'{predict_file} not exists')
            return None
        match_points_cnt_arr = np.loadtxt(predict_file, dtype=int)

        val_info = np.loadtxt(val_file, dtype=str, delimiter=',')
        val_info = [[x.strip() for x in line] for line in val_info]
        val_info = np.array(val_info)
        gt_arr = val_info[:, 2].astype(int)

        scaled_scores = match_points_cnt_arr / max(match_points_cnt_arr)
        precision, recall, thresholds = precision_recall_curve(gt_arr, scaled_scores)
        average_precision = average_precision_score(gt_arr, scaled_scores)

        # max recall on 100% precision
        max_recall = max(recall[precision == 1])
        result[scene] = (precision, recall, thresholds, average_precision, max_recall)
        result[scene + '_match_points'] = match_points_cnt_arr
    return result


# load results
all_results = {}
for version_str in all_version_str_arr:
    result = get_version_result(version_str)
    if result is not None:
        all_results[version_str] = result

plot_version_str_arr = ['SIFT', 'ORB', 'Aspanformer', 'Aspanformer_RemoveEgo',
                        'Aspanformer_RemoveEgo_BgMask'
                        ]
color_arr = ['tab:blue', 'tab:orange', 'tab:purple', 'tab:green', 'tab:red', ]

# align
plot_version_name_str_arr = ['SIFT                                       ',
                             'ORB                                       ',
                             'Aspan                                    ',
                             'Aspan+RemoveEgo               ',
                             'Aspan+RemoveEgo+BgMask']

# plot
plot_out_dir = os.path.join(save_dir, 'plot')
os.makedirs(plot_out_dir, exist_ok=True)

for scene in scenes:
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(len(plot_version_str_arr)):
        version_str = plot_version_str_arr[i]
        version_name_str = plot_version_name_str_arr[i]
        color = color_arr[i]
        if version_str not in all_results:
            continue
        precision, recall, thresholds, average_precision, max_recall = all_results[version_str][scene]
        result_auc = auc(recall, precision)
        label_str = f'{version_name_str} AUC={result_auc:.2f} MR@100P={max_recall:.2f}'
        ax.plot(recall, precision, label=label_str, color=color)

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    scene_str = scene[9:-6]
    ax.set_title(f'Precision-Recall Curve on {scene}')
    ax.legend()
    fig.savefig(f'{plot_out_dir}/pr_curve_{scene}.png')
    plt.close(fig)

print('Done! Plot saved in ', os.path.abspath(plot_out_dir))
