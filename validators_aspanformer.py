import os
import sys
import numpy as np
import cv2
import torch

ASPANFORMER_DIR = 'ml-aspanformer-main'
sys.path.insert(0, ASPANFORMER_DIR)

from src.ASpanFormer.aspanformer import ASpanFormer
from src.config.default import get_cfg_defaults
from src.utils.misc import lower_config

from utils import draw_match

config_path = 'ml-aspanformer-main/configs/aspan/outdoor/aspan_test.py'
weights_path = './weights/outdoor.ckpt'

config = get_cfg_defaults()
config.merge_from_file(config_path)
_config = lower_config(config)
matcher = ASpanFormer(config=_config['aspan'])
state_dict = torch.load(weights_path, map_location='cpu')['state_dict']
matcher.load_state_dict(state_dict, strict=False)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
matcher.to(device)
matcher.eval()


def match_by_aspanformer(img0_path, img1_path, remove_ego=False):
    img0_g, img1_g = cv2.imread(img0_path, 0), cv2.imread(img1_path, 0)
    if remove_ego:
        img0_g = img0_g[:int(img0_g.shape[0] * 0.83), :]
        img1_g = img1_g[:int(img1_g.shape[0] * 0.83), :]
    data = {'image0': torch.from_numpy(img0_g / 255.)[None, None].float().to(device),
            'image1': torch.from_numpy(img1_g / 255.)[None, None].float().to(device), }
    with torch.no_grad():
        matcher(data, online_resize=True)
        corr0, corr1 = data['mkpts0_f'].cpu().numpy(), data['mkpts1_f'].cpu().numpy()
    return corr0, corr1


def find_fundamental_mat(corr0, corr1):
    F_hat, mask_F = cv2.findFundamentalMat(corr0, corr1, method=cv2.FM_RANSAC, ransacReprojThreshold=1)
    if mask_F is not None:
        mask_F = mask_F[:, 0].astype(bool)
    else:
        mask_F = np.zeros_like(corr0[:, 0]).astype(bool)
    return F_hat, mask_F


# ransac
def loop_verification_aspanformer(img0_path, img1_path):
    corr0, corr1 = match_by_aspanformer(img0_path, img1_path, remove_ego=False)

    F_hat, mask_F = find_fundamental_mat(corr0, corr1)

    # visualize match
    # img0, img1 = cv2.imread(img0_path), cv2.imread(img1_path)
    # display = draw_match(img0, img1, corr0, corr1)
    # display_ransac = draw_match(img0, img1, corr0[mask_F], corr1[mask_F])
    # cv2.imwrite('match.png', display)
    # cv2.imwrite('match_ransac.png', display_ransac)

    # print(len(corr1), len(corr1[mask_F]))
    return corr0[mask_F], corr1[mask_F]


# remove car ego
def loop_verification_aspanformer_remove_ego(img0_path, img1_path):
    corr0, corr1 = match_by_aspanformer(img0_path, img1_path, remove_ego=True)
    F_hat, mask_F = find_fundamental_mat(corr0, corr1)
    return corr0[mask_F], corr1[mask_F]


def img_path_to_mask_path(img_path):
    # example:
    # img_path: /path/to/data/Autumn_mini_query/000000.jpg
    # mask_path: /path/to/data/Autumn_mini_query_mask/000000.npy
    dataset_name = os.path.basename(os.path.dirname(img_path))
    img_name = os.path.basename(img_path).split('.')[0]
    dataset_root = os.path.abspath(os.path.join(img_path, "../.."))
    depth_path = os.path.join(dataset_root, dataset_name + '_mask', img_name + '.npy')
    return depth_path


def load_mask_from_img_path(img_path):
    mask_path = img_path_to_mask_path(img_path)
    mask = np.load(mask_path)
    return mask


def loop_verification_aspanformer_remove_ego_bgmask(img0_path, img1_path):
    corr0, corr1 = match_by_aspanformer(img0_path, img1_path, remove_ego=True)

    background_mask0 = load_mask_from_img_path(img0_path)
    background_mask1 = load_mask_from_img_path(img1_path)

    corr0_mask = background_mask0[corr0[:, 0].astype(int), corr0[:, 1].astype(int)]
    corr1_mask = background_mask1[corr1[:, 0].astype(int), corr1[:, 1].astype(int)]

    mask_use = corr0_mask * corr1_mask
    mask_use = mask_use.astype(bool)

    corr0_filtered = corr0[mask_use]
    corr1_filtered = corr1[mask_use]

    F_hat, mask_F = find_fundamental_mat(corr0_filtered, corr1_filtered)

    # F_hat_ransac, mask_F_ransac = find_fundamental_mat(corr0, corr1)
    # print(len(corr1), len(corr0[mask_F_ransac]), len(corr0_filtered[mask_F]))

    return corr0_filtered[mask_F], corr1_filtered[mask_F]
