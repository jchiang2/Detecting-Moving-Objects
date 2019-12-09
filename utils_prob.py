import cv2
import numpy as np
from collections import defaultdict
from utils_patches import calcArea

def build_probability_map():
    # TODO
    pass

def build_corr_map(img, patch_groups, match_groups):
    # TODO
    largestDist = getLargestDist(img, patch_groups) / 9 * 2
    # input(largestDist)

    imgH, imgW = img.shape[:2]
    corr_map = np.zeros((imgH,imgW))
    img = np.zeros(corr_map.shape)
    # flat_patch_groups = [item for sublist in patch_groups for item in sublist]
    # flat_match_groups = [item for sublist in match_groups for item in sublist]
    print("Patch groups: ", len(patch_groups))
    print("Match groups: ", len(match_groups))
    for ind, (patches, matches) in enumerate(zip(patch_groups, match_groups)):
        if len(patches) == 0 or len(matches) == 0:
            continue
        for (patch, match) in zip(patches, matches):
        # print("Patch:{}, Match:{}".format(ind, match),  end="\r", flush=True)

            patch_center = np.sum(patch, axis=0) / patch.shape[0]
            
            patch = np.array(patch, np.int32)
            coor_rows, coor_cols = polyCoors(patch)
            
            dist_rows = (coor_rows - patch_center[1])**2
            dist_cols = (coor_cols - patch_center[0])**2
            dist = dist_rows + dist_cols

            dist = np.exp(-dist / largestDist)

            corr_map[coor_rows, coor_cols] += dist * match
            # corr_map[coor_rows, coor_cols] += match
            # corr_map[coor_rows, coor_cols] += 1

            # cv2.imshow("Frame", corr_map)
            # k = cv2.waitKey(0)
            # if k==27:    # Esc key to stop
            #     break

    return corr_map

def getLargestDist(img, patch_groups):
    flat_patch_groups = [item for sublist in patch_groups for item in sublist]
    imgH, imgW = img.shape[:2]
    corr_map = np.zeros((imgH,imgW))
    largestDist = 0
    for ind, patch in enumerate(flat_patch_groups):
        if not patch.size:
            continue
        print("Patch:", ind, end="\r", flush=True)

        patch_center = np.sum(patch, axis=0) / patch.shape[0]

        patch = np.array(patch, np.int32)
        coor_rows, coor_cols = polyCoors(patch)
        
        dist_rows = (coor_rows - patch_center[1])**2
        dist_cols = (coor_cols - patch_center[0])**2
        dist = dist_rows + dist_cols
        maxDist = np.amax(dist)
        largestDist = max(maxDist, largestDist)
    return largestDist

def polyCoors(pts):
    '''
    Return pixel coordinates inside polygon defined by the given points
    Arg
        pts: numpy array of polygon vertices (n, 2)
    Return
        row coordinates (n,)
        col coordinates (n,)
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
    coor_rows = coors[0] + offset[1]
    coor_cols = coors[1] + offset[0]

    return coor_rows, coor_cols
