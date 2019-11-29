import cv2
import numpy as np
import os

def detectMatchingPoints(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    good = []
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    return pts1, pts2

def calcFundamental(pts1, pts2):
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]
    return F, pts1, pts2

def getEpilines(F, pts1, pts2):
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2, F)
    lines1 = lines1.reshape(-1,3)
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1, F)
    lines2 = lines2.reshape(-1,3)
    return lines1, lines2

def loadData(img1, img2, save_dir):
    outfile = os.path.join(save_dir, "epipolar_data.npz")
    if (os.path.exists(outfile)):
        data = np.load(outfile)
        F, pts1, pts2, lines1, lines2 = data['F'], data['pts1'], data['pts2'], data['lines1'], data['lines2']
    else:
        pts1, pts2 = detectMatchingPoints(img1, img2)
        F, pts1, pts2 = calcFundamental(pts1, pts2)
        lines1, lines2 = getEpilines(F, pts1, pts2)
        np.savez(outfile, F=F, pts1=pts1, pts2=pts2, lines1=lines1, lines2=lines2)
    return F, pts1, pts2, lines1, lines2

def getEpipole(lines):
    epipole = np.cross(lines[0], lines[1])
    epipole = epipole / epipole[2]
    return epipole