import cv2
import numpy as np


def loop_verification_SIFT(img0_path, img1_path):
    img1 = cv2.imread(img0_path)
    img2 = cv2.imread(img1_path)

    # 提取关键点和描述符
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 进行特征匹配并进行几何验证
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    pts1 = []
    pts2 = []
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)
            good_matches.append([m])

    pts1 = np.array(pts1)
    pts2 = np.array(pts2)

    # Compute fundamental matrix
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=0.5, confidence=0.99)

    # We select only inlier points
    if mask is None:
        return np.empty((0, 2)), np.empty((0, 2))
    else:
        return pts1[mask.ravel() == 1], pts2[mask.ravel() == 1]


def loop_verification_ORB(img0_path, img1_path):
    # 读取两幅图像
    img1 = cv2.imread(img0_path)
    img2 = cv2.imread(img1_path)
    # 提取关键点和描述符
    orb = cv2.ORB_create()
    orb.setScoreType(cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
    orb.setWTA_K(3)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    # 暴力匹配
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    # extract points
    pts1 = []
    pts2 = []
    good_matches = []
    for i, (m) in enumerate(matches):
        if m.distance < 20:
            # print(m.distance)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            good_matches.append(matches[i])

    pts1 = np.array(pts1)
    pts2 = np.array(pts2)

    # Compute fundamental matrix
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=0.5, confidence=0.99)

    # We select only inlier points
    if mask is None:
        return np.empty((0, 2)), np.empty((0, 2))
    else:
        return pts1[mask.ravel() == 1], pts2[mask.ravel() == 1]
