import cv2
import numpy as np
import os
import argparse

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Visualize epipolar lines")
    parser.add_argument(
        "--config-file", help="Path to config file", default="configs/default.yaml", type=str)
    return parser.parse_args()

def drawLines(img1, img2, lines1, lines2):
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)

    for line1, line2 in zip(lines1, lines2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -line1[2]/line1[1]])
        x1,y1 = map(int, [c, -(line1[2]+line1[0]*c)/line1[1]])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color, 10)
        x0,y0 = map(int, [0, -line2[2]/line2[1]])
        x1,y1 = map(int, [c, -(line2[2]+line2[0]*c)/line2[1]])
        img2 = cv2.line(img2, (x0,y0), (x1,y1), color, 10)
    return img1, img2

def drawPatches(img, patches):
    for patch in patches:
        color = tuple(np.random.randint(0,255,3).tolist())

        patch = np.array(patch, np.int32)
        patch = patch.reshape((-1,1,2))
        cv2.fillPoly(img,[patch],color)

    return img

def padd_patches(img, pts, h, w):
    '''
    Padd cropped patch
    Args
        img: target gray scale image that will be cropped (h, w)
        pts: numpy array of patch coordinates (4, 2)
        h: the padding height (>= than cropped image height)
        w: the padding width (>= than cropped image width)
    Return
        cropped and padded patch
    '''
    padded = np.zeros((h, w), np.uint8)
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    cropped = img[y:y+h, x:x+w].copy()

    # make mask
    pts = pts - pts.min(axis=0)

    mask = np.zeros(cropped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (1, 1, 1), -1, cv2.LINE_AA)

    # do bit-op
    masked = cv2.bitwise_and(cropped, cropped, mask=mask)
    padded[0:masked.shape[0], 0:masked.shape[1]] = masked

#     cv2.imshow('padded', padded)
#     cv2.waitKey(10)
    return padded

# coor = np.array([[600, 500],[600, 1000],[1300,1500],[1500, 200]])
# padd_patches(img1, coor, 1500, 1000)
