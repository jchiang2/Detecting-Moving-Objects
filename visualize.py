from matplotlib import pyplot as plt
import cv2
import numpy as np
import os
import math

from utils import *
from utils_epipolar import *
from utils_patches import *
from utils_hog import *
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

    ## Load images
    images = os.listdir(img_dir)
    im_path1 = os.path.join(img_dir, images[3])
    im_path2 = os.path.join(img_dir, images[4])
    print("Loading image:", im_path1)
    print("Loading image:", im_path2)
    img1 = cv2.imread(im_path1,0) # reference image # left image
    img2 = cv2.imread(im_path2,0) # support image # right image


    ## Calculate matching feature points and epipoles
    F, pts1, pts2, epipole1, epipole2 = loadData(cfg, im_path1, im_path2)
    # epipole1 = getEpipole(lines1)
    # epipole2 = getEpipole(lines2)


    ## Get radial lines (equally spaced) crossing epipole
    numLines = int(180 / angle)
    if numLines % 2 == 0:  # Even number of lines results in division by zero
        numLines += 1
    lines1, radialPts = getRadialLines(numLines, epipole1)
    lines1, valid_pts = find_valid_lines(img1, lines1, radialPts)
    lines2 = find_corr_lines(valid_pts, F)


    ## Visualize lines
    # img1, img2 = drawLines(img1, img2, lines1, lines2)


    ## Get vertices along lines
    img1, patch_points1 = getVertices(img1, epipole1, lines1, height)
    img2, patch_points2 = getVertices(img2, epipole2, lines2, height)


    ## Group vertices into patches
    patch_groups1 = getPatches(cfg, img1, patch_points1)
    patch_groups2 = getPatches(cfg, img2, patch_points2, isSupport=True)

    ## Perform HOG feature matching for patches
    matchPatches(img1, img2, patch_groups1, patch_groups2)

    ## Visualize all patches
    # img1 = drawPatches(img1, patch_groups1)
    # img2 = drawPatches(img2, patch_groups2)
    
    plt.subplot(121),plt.imshow(img1)
    plt.subplot(122),plt.imshow(img2)
    plt.show()

    return

if __name__ == "__main__":
    args = parse_args()
    main(args)