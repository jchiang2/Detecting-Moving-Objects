from matplotlib import pyplot as plt
import cv2
import numpy as np
import os
import math

from utils import *
from utils_epipolar import *
from utils_patches import *
from config import _C as cfg

PI = math.pi

def main(args):
    cfg.merge_from_file(args.config_file)
    img_dir = cfg.IMAGE_PATH
    save_dir = cfg.SAVE_PATH
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    height = cfg.HEIGHT
    angle = cfg.ANGLE
    overlap = cfg.OVERLAP

    images = os.listdir(img_dir)
    im_path1 = os.path.join(img_dir, images[0])
    im_path2 = os.path.join(img_dir, images[1])

    img1 = cv2.imread(im_path1,0) # queryimage # left image
    img2 = cv2.imread(im_path2,0) # trainimage # right image

    # Calculate matching points and epipoles
    F, pts1, pts2, lines1, lines2 = loadData(img1, img2, save_dir)
    epipole1 = getEpipole(lines1)
    epipole2 = getEpipole(lines2)

    # Get radial lines from epipole
    numLines = int(360 / angle)
    if numLines % 2 == 0:
        numLines += 1
    lines1, lines2, radialPts = getRadialLines(numLines, epipole1, F)
    lines1 = find_valid_lines(lines1, img1)

    # Visualize lines
    img1, img2 = drawLines(img1, img2, lines1, lines2)
    
    # Get points along lines and visualize
    img1, points = getPoints(img1, epipole1, lines1, 100)

    # Sample points into patches
    patches = getPatches(cfg, img1, points)

    img1 = drawPatches(img1, patches)

    plt.subplot(121),plt.imshow(img1)
    plt.subplot(122),plt.imshow(img2)
    plt.show()

    return

if __name__ == "__main__":
    args = parse_args()
    main(args)