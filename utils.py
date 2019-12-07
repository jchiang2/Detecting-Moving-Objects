import cv2
import numpy as np
import os
import argparse

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Visualize epipolar lines")
    parser.add_argument(
        "--config-file", help="Path to config file", default="configs/default.yaml", type=str)
    return parser.parse_args()

def drawLines(img1, img2, lines1, lines2):
    '''
    Visualize lines on images.
    Args:
        img1: (numpy array h x w) Reference image (grayscale)
        img2: (numpy array h x w) Support image (grayscale)
        lines1: (list N x 3) List of N lines in image 1 
        lines2: (list N x 3) List of N corresponding lines in image 2
    returns:
        img1: Reference image with lines
        img2: Support image with lines
    '''
    r,c = img1.shape
    img1 = img1.copy()
    img2 = img2.copy()
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)

    for i, (line1, line2) in enumerate(zip(lines1, lines2)):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -line1[2]/line1[1]])
        x1,y1 = map(int, [c, -(line1[2]+line1[0]*c)/line1[1]])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color, 10)
        x0,y0 = map(int, [0, -line2[2]/line2[1]])
        x1,y1 = map(int, [c, -(line2[2]+line2[0]*c)/line2[1]])
        img2 = cv2.line(img2, (x0,y0), (x1,y1), color, 10)
    return img1, img2

def drawPatches(img, patch_groups):
    '''
    Visualize patches on image.
    Args:
        img: (numpy array h x w x c) Color image to draw patches on
        patch_groups: (list N x M x 4 x 2) Patch groups for N pairs of epipolar lines, 
                      each with M patches
    returns:
        img: Image with patches
    '''
    if len(img.shape) == 2:
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    for patches in patch_groups:
        for patch in patches:
            color = tuple(np.random.randint(0,255,3).tolist())

            patch = np.array(patch, np.int32)
            patch = patch.reshape((-1,1,2))
            cv2.fillPoly(img,[patch],color)
    return img

def drawPatch(img, patch):
    '''
    Visualize a single given patch.
    Args:
        img: (numpy array h x w x c) Color image to draw patch on
        patch: (list 4 x 2) A single patch represented by four 2D vertice coordinates
    returns:
        img: Image with patch
    '''
    color = tuple(np.random.randint(0,255,3).tolist())
    patch = np.array(patch, np.int32)
    patch = patch.reshape((-1,1,2))
    cv2.fillPoly(img,[patch],color)
    return img
