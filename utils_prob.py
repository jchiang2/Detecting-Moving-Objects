import cv2
import numpy as np
from collections import defaultdict
from utils_patches import calcArea

def build_probability_map():
    # TODO
    pass

def build_corr_map(img, patch_groups, match_groups):
    # TODO

    imgH, imgW = img.shape[:2]
    corr_map = np.zeros((imgH,imgW))
    img = np.zeros(corr_map.shape)
    flat_patch_groups = [item for sublist in patch_groups for item in sublist]
    flat_match_groups = [item for sublist in match_groups for item in sublist]
    for ind, (patch, match) in enumerate(zip(flat_patch_groups, flat_match_groups)):
        print("Patch:", ind, end="\r", flush=True)

        patch_center = np.sum(patch, axis=0) / patch.shape[0]

        patch = np.array(patch, np.int32)
        x,y,w,h = cv2.boundingRect(patch)
        patch -= np.array([x, y])

        patch = patch.reshape((-1,1,2))
        cropped = img[y:y+h, x:x+w].copy()
        cv2.fillPoly(cropped,[patch],1)

        pixels = np.argwhere(cropped==1) + np.array([x, y])
        pixels = pixels[(pixels[:,0] < imgH) & (pixels[:,1] < imgW)]
        dist = np.exp(-np.sum((pixels - patch_center)**2, axis=1))
        corr_map[pixels[:,0],pixels[:,1]] += dist * match
    return corr_map