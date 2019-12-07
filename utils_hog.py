import cv2
import numpy as np
from utils import drawPatch

def pad_patch(img, pts, h, w):
    '''
    Pad cropped patch
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

def getLargestBound(patches):
    '''
    Retrieves the largest heigth and width in a set of patches.
    Args:
        patches: (list N x 4 x 2) List of patches
    returns:
        h_max: Largest height
        w_max: Largest width
    '''
    w_max = 0
    h_max = 0
    for i, patch in enumerate(patches):
        patch = np.array(patch)
        patch = patch.astype(np.int32)
        x, y, w, h = cv2.boundingRect(patch)
        w_max = max(w_max, w)
        h_max = max(h_max, h)
    return h_max, w_max

def getHOGDescriptor(img, patch_group, h, w):
    '''
    Calculates the HOG descriptor for a set of patches.
    Args:
        img: (numpy array h x w x 3) Origin color image
        patch_group: (list N x 4 x 2) List of patches
        h: (int) Descriptor window height
        w: (int) Descriptor window width
    returns:
        descriptor_set: (list N x K) List of N descriptors of length K
    '''
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).copy()
    hog = cv2.HOGDescriptor()

    descriptor_set = []
    print("Number of patches:", len(patch_group))
    for i, patch in enumerate(patch_group):
        patch = np.array(patch)
        patch = patch.astype(np.int32)
        padded = pad_patch(img, patch, h, w)
        descriptor = hog.compute(padded)
        print("Patch {}:".format(i), end="\r", flush=True)
        descriptor_set.append(descriptor[:,0])

    return descriptor_set

def matchPatches(img1, img2, patch_groups1, patch_groups2):
    '''
    Matches each patch in the reference image to its closest patch in the support
    image using cosine similarity.
    Args:
        img1(img2): (numpy array h x w x 3) Reference(support) image
        patch_groups1(patch_groups2): (list N x M x 4 x 2) Patch groups for N pairs
                                      of epipolar lines, each with M patches
    return:
        match_groups: (numpy array N x M) Indices of corresponding patches in support image
    '''
    match_groups = []
    for group1, group2 in zip(patch_groups1, patch_groups2):
        h, w = getLargestBound(group1 + group2)
        hog1 = getHOGDescriptor(img1, group1, h, w)
        hog2 = getHOGDescriptor(img2, group2, h, w)
        hog1 = hog1 / np.linalg.norm(hog1, axis=1, keepdims=True)
        hog2 = hog2 / np.linalg.norm(hog2, axis=1, keepdims=True)
        cos_sim = np.matmul(hog1, hog2.T)
        matches = np.argmax(cos_sim, axis=1)
        # for i, hog_ref in enumerate(hog1):
        #     cos_sim = hog_ref * hog2
        #     cos_sim = np.linalg.norm(hog_ref)np.linalg.norm(hog2, axis=1)
        #     match = np.argmax(cosine)

        #     # Visualize matching
        #     color = tuple(np.random.randint(0,255,3).tolist())
        #     drawPatch(img1, group1[i])
        #     drawPatch(img2, group2[match])
        #     concat = np.concatenate((img1, img2), axis=1)
        #     cv2.imshow("Frame", concat)
        #     k = cv2.waitKey(10) & 0xFF
        match_groups.append(matches)
    return match_groups
        