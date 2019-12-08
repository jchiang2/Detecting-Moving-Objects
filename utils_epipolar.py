import cv2
import numpy as np
import os

def detectMatchingPoints(img1, img2):
    '''
    Get matching SIFT feature points in two images.
    Args:
        img1: (numpy array h x w) Reference image in grayscale
        img2: (numpy array h x w) Support image in grayscale
    returns:
        pts1: (list N x 2) List of N matching points in the reference image
        pts2: (list N x 2) List of N matching points in the support image
    '''
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
    '''
    Calculates fundamental matrix.
    Args:
        pts1: (list N x 2) N matching points in reference image
        pts2: (list N x 2) N matching points in support image
    returns:
        F: (numpy array 3 x 3) Fundamental matrix
        pts1_inlier: (numpy array M x 2) Inliers in reference image
        pts2_inlier: (numpy array M x 2) Inliers in support image
    '''
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
    # We select only inlier points
    pts1_inlier = pts1[mask.ravel()==1]
    pts2_inlier = pts2[mask.ravel()==1]
    return F, pts1_inlier, pts2_inlier

def getEpipoles(F, pts1, pts2):
    '''
    Calculates coordinates of epipoles.
    Args:
        F: (numpy array 3 x 3) Fundamental matrix
        pts1: (numpy array M x 2) Inliers in reference image
        pts2: (numpy array M x 2) Inliers in support image
    returns:
        epipole1: (numpy array 1 x 3) Epipole in reference image in homogenous coordinates
        epipole2: (numpy array 1 x 3) Epipole in support image in homogenous coordinates
    '''
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2, F)
    lines1 = lines1.reshape(-1,3)
    epipole1 = np.cross(lines1[0], lines1[1])
    epipole1 = epipole1 / epipole1[2]
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1, F)
    lines2 = lines2.reshape(-1,3)
    epipole2 = np.cross(lines2[0], lines2[1])
    epipole2 = epipole2 / epipole2[2]
    return epipole1, epipole2

def epipoleSVD(M):
    V = cv2.SVDecomp(M)[2]
    return V[-1]/V[-1,-1]

def loadData(cfg, img1, img2, outfile=None, debug=False):
    '''
    Loads epipolar data if available, otherwise calls operations to get data
    Args:
        cfg: Config class
        img1: (numpy array h x w) Reference image in grayscale
        img2: (numpy array h x w) Support image in grayscale
    returns:
        F: (numpy array 3 x 3) Fundamental matrix
        pts1: (numpy array N x 2) Matching points in reference image
        pts2: (numpy array N x 2) Matching points in support image
        epipole1: (numpy array 1 x 3) Epipole in reference image in homogenous coordinates
        epipole2: (numpy array 1 x 3) Epipole in support image in homogenous coordinates
    '''
    if (os.path.exists(outfile) and not debug):  # Load data if available
        data = np.load(outfile)
        F, pts1, pts2, epipole1, epipole2 = data['F'], data['pts1'], data['pts2'], data['epipole1'], data['epipole2']
    else:  # Calculate data
        pts1, pts2 = detectMatchingPoints(img1, img2)
        F, pts1, pts2 = calcFundamental(pts1, pts2)
        epipole1, epipole2 = getEpipoles(F, pts1, pts2)
        np.savez(outfile, F=F, pts1=pts1, pts2=pts2, epipole1=epipole1, epipole2=epipole2)
    return F, pts1, pts2, epipole1, epipole2
