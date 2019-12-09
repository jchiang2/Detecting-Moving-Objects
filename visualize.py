from matplotlib import pyplot as plt
import cv2
import numpy as np
import os
import math
import random

from utils import *
from utils_epipolar import *
from utils_patches import *
from utils_hog import *
from utils_prob import *
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
    images = sorted(os.listdir(img_dir))
    ref = cfg.REFERENCE
    sup_imgs = cfg.SUPPORT
    # IM1, IM2 = random.sample(range(len(images)), 2)
    # IM1 = 10
    # IM2 = 11
    

    for sup in sup_imgs:
        im_path1 = os.path.join(img_dir, images[ref])
        im_path2 = os.path.join(img_dir, images[sup])
        print("=========================================")
        print("Using reference image {}".format(im_path1))
        print("Running on support image {}".format(im_path2))
        print("=========================================")
        # print("Loading image:", im_path1)
        # print("Loading image:", im_path2)
        img1 = cv2.imread(im_path1,0) # reference image # left image
        img2 = cv2.imread(im_path2,0) # support image # right image
        if cfg.RESIZE:
            img1 = resize(cfg, img1)
            img2 = resize(cfg, img2)

        # Epipolar data
        ind1 = int(os.path.splitext(images[ref])[0])
        ind2 = int(os.path.splitext(images[sup])[0])
        outfile = os.path.join(cfg.SAVE_PATH, "epipolar_data_{}_{}_{}.npz".format(os.path.basename(img_dir), ind1, ind2))
        ## Calculate matching feature points and epipoles
        F, pts1, pts2, epipole1, epipole2 = loadData(cfg, img1, img2, outfile, debug=False)
        
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
        patch_points1 = getVertices(img1, epipole1, lines1, height)
        patch_points2 = getVertices(img2, epipole2, lines2, height)

        ## Group vertices into patches
        patch_groups1 = getPatches(cfg, img1, patch_points1)
        patch_groups2 = getPatches(cfg, img2, patch_points2, isSupport=True)

        ## Perform HOG feature matching for patches
        match_file = os.path.join(cfg.SAVE_PATH, "matches_{}_{}_{}.pkl".format(os.path.basename(img_dir), ind1, ind2))
        if os.path.isfile(match_file):
            with open(match_file, 'rb') as f:
                match_groups = pickle.load(f)
        else:
            match_groups = matchPatches(img1, img2, patch_groups1, patch_groups2)
            with open(match_file, 'wb') as output:
                pickle.dump(match_groups, output, -1)

        ## Build pixel correspondence map
        corr_map = build_corr_map(img1, patch_groups1, match_groups)
        # corr_map = cv2.normalize(corr_map, corr_map, 0, 255, cv2.NORM_MINMAX)

        ## Visualize all patches
        # img1 = drawPatches(img1, patch_groups1)
        # img2 = drawPatches(img2, patch_groups2)

        np.save(os.path.join(cfg.SAVE_PATH, "prob_map_{}_{}_{}.npy".format(os.path.basename(img_dir), ind1, ind2)), corr_map)

        # plt.subplot(131),plt.imshow(img1)
        # plt.subplot(132),plt.imshow(img2)
        # plt.subplot(133),plt.imshow(corr_map)
        # plt.show()

    return

if __name__ == "__main__":
    args = parse_args()
    main(args)