import cv2
import numpy as np
from collections import defaultdict
from utils_patches import calcArea

def build_probability_map():
    # TODO
    pass

def build_corr_map(img, patch_groups, match_groups):
    # TODO
    # largestDist = getLargestDist(img, patch_groups)

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
        patch -= np.array([y, x])

        patch = patch.reshape((-1,1,2))
        cropped = img[y:y+h, x:x+w].copy()
        cv2.fillPoly(cropped,[patch],1)

        pixels = np.argwhere(cropped==1) + np.array([y, x])
        pixels = pixels[(pixels[:,0] < imgH) & (pixels[:,1] < imgW)]
        dist = np.exp(-np.sum((pixels - patch_center)**2, axis=1))
        print(dist)
        # corr_map[pixels[:,0],pixels[:,1]] += dist * match
        corr_map[pixels[:,0],pixels[:,1]] += match

    return corr_map

def getLargestDist(img, patch_groups):
    flat_patch_groups = [item for sublist in patch_groups for item in sublist]
    imgH, imgW = img.shape[:2]
    corr_map = np.zeros((imgH,imgW))
    largestDist = 0
    for ind, (patch, match) in enumerate(flat_patch_groups):
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
        dist = np.sum((pixels - patch_center)**2, axis=1)
        print(dist)
        maxDist = np.amax(dist)
        largestDist = max(maxDist, largestDist)
        # corr_map[pixels[:,0],pixels[:,1]] += dist * match
    return largestDist

def polyCoors(pts):
    '''
    Return pixel coordinates inside polygon defined by the given points
    Arg
        pts: numpy array of polygon vertices (n, 2)
    Return
        x coordinates (n,)
        y coordinates (n,)
    '''
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    mask = np.zeros((h, w), np.uint8)
    offset = pts.min(axis=0)
#    print(offset)
    pts = pts - offset
    cv2.drawContours(mask, [pts], -1, (1, 1, 1), -1, cv2.LINE_AA)
#     plt.figure()
#     plt.imshow(mask)
#     plt.show()
    coors = np.where(mask == 1)
    coors_x = coors[0] + offset[1]
    coors_y = coors[1] + offset[0]

    return coors_x, coors_y
