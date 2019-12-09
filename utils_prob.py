import cv2
import numpy as np
from collections import defaultdict
from utils_patches import calcArea

def build_probability_map():
    # TODO
    pass

def build_corr_map(img, patch_groups, match_groups):
    # TODO
    # largestDist = getLargestDist(img, patch_groups) / 9 * 2
    # input(largestDist)
    largestDist = 7670.238592346108

    imgH, imgW = img.shape[:2]
    corr_map = np.zeros((imgH,imgW))
    img = np.zeros(corr_map.shape)
    # flat_patch_groups = [item for sublist in patch_groups for item in sublist]
    # flat_match_groups = [item for sublist in match_groups for item in sublist]
    for ind, (patches, matches) in enumerate(zip(patch_groups, match_groups)):
        for (patch, match) in zip(patches, matches):
        # print("Patch:{}, Match:{}".format(ind, match),  end="\r", flush=True)

            patch_center = np.sum(patch, axis=0) / patch.shape[0]
            
            patch = np.array(patch, np.int32)
            x,y,w,h = cv2.boundingRect(patch)
            patch -= np.array([x, y])

            patch = patch.reshape((-1,1,2))
            cropped = img[y:y+h, x:x+w].copy()
            cv2.fillPoly(cropped,[patch],1)

            pixels = np.argwhere(cropped==1) + np.array([y, x])
            # pixels = pixels[(pixels[:,0] < imgH) & (pixels[:,1] < imgW)]
            # dist = np.exp(-np.sum((pixels - patch_center[::-1])**2, axis=1) / largestDist)
            # corr_map[pixels[:,0],pixels[:,1]] += dist * match
            corr_map[pixels[:,0],pixels[:,1]] += match

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
        x,y,w,h = cv2.boundingRect(patch)
        patch -= np.array([x, y])

        patch = patch.reshape((-1,1,2))
        cropped = corr_map[y:y+h, x:x+w].copy()
        cv2.fillPoly(cropped,[patch],1)

        pixels = np.argwhere(cropped==1) + np.array([y, x])
        pixels = pixels[(pixels[:,0] < imgH) & (pixels[:,1] < imgW)]

        if not pixels.size:
            continue

        dist = np.sum((pixels - patch_center[::-1])**2, axis=1)
        maxDist = np.amax(dist)
        largestDist = max(maxDist, largestDist)
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
