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