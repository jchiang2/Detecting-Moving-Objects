import cv2
import numpy as np
import scipy.ndimage
import os
import pickle
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


    # HOG Parameters
    # hog = cv2.HOGDescriptor()
    cellRes = 8
    blockRes = 2
    hog = cv2.HOGDescriptor(_winSize=(w//cellRes * cellRes, h//cellRes * cellRes),
                        _blockSize=(w//cellRes * blockRes, h//cellRes * blockRes),
                        _blockStride=(w//cellRes, h//cellRes),
                        _cellSize=(w//cellRes, h//cellRes),
                        _nbins=18)
    # ASSERT
    # (winSize.width - blockSize.width) % blockStride.width == 0 and
    # (winSize.height - blockSize.height) % blockStride.height == 0

    # img = computeHOG(img)

    descriptor_set = []
    # print("Number of patches:", len(patch_group))
    for i, patch in enumerate(patch_group):
        patch = np.array(patch)
        patch = patch.astype(np.int32)
        padded = pad_patch(img, patch, h, w)
        descriptor = hog.compute(padded)
        # print("Patch {}:".format(i), end="\r", flush=True)
        descriptor_set.append(descriptor[:,0])
    descriptor_set = np.array(descriptor_set)
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
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR).copy()
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR).copy()

    # img1 = computeHOG(img1)
    # img2 = computeHOG(img2)

    match_groups = []
    for i, (group1, group2) in enumerate(zip(patch_groups1, patch_groups2)):
        if len(group1) == 0 or len(group2) == 0:
            match_groups.append([])
            continue
        print("Line: ", i, end='\r', flush=True)
        h, w = getLargestBound(group1 + group2)
        hog1 = getHOGDescriptor(img1, group1, h, w)
        hog2 = getHOGDescriptor(img2, group2, h, w)


        hog1 = hog1 / np.linalg.norm(hog1, axis=1, keepdims=True)
        hog2 = hog2 / np.linalg.norm(hog2, axis=1, keepdims=True)
        cos_sim = np.matmul(hog1, hog2.T)
        # matches = np.argmax(cos_sim, axis=1)
        matches = np.amax(cos_sim, axis=1)


        # print(hog1.shape)
        # print(hog2.shape)
        # print("Computing distance...")
        # N = hog1.shape[0]
        # M = hog2.shape[0]
        # hog1 = np.repeat(hog1, M, axis=0)
        # print("Hog1 done")
        # hog2 = np.tile(hog2, (N, 1))
        # print("Hog2 done")

        # matches = np.linalg.norm(hog1-hog2, axis=1)
        # matches = np.reshape(matches, (M, -1))
        # matches = np.argmax(matches, axis=0)

        # print(matches)

        match_groups.append(matches)
        # match_groups.append(cos_sim)
    
    return match_groups

def computeHOG(img, cell_size=(4, 4), block_size=(2, 2), n_bin=9):
    '''
    Compute HOG features and output as a image form
    Arg
        img: input image (W, H, 3)
        cell_size: tuple of desired cell size (int, int) in pixels
        block_size: tuple of desired block size (int, int) in number of cells
        n_bin: number of orientation bins
    Return
        Computed HOG descriptor (imgH // cell[1], imgW // cell[0], n_bin)
    '''
    print("Computing HOG image...")
    # winSize is the size of the image cropped to an multiple of the cell size
    hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                  img.shape[0] // cell_size[0] * cell_size[0]),
                        _blockSize=(block_size[1] * cell_size[1],
                                    block_size[0] * cell_size[0]),
                        _blockStride=(cell_size[1], cell_size[0]),
                        _cellSize=(cell_size[1], cell_size[0]),
                        _nbins=n_bin)

    n_cells = (img.shape[0] // cell_size[0], img.shape[1] // cell_size[1])
    hog_feats = hog.compute(img)\
               .reshape(n_cells[1] - block_size[1] + 1,
                        n_cells[0] - block_size[0] + 1,
                        block_size[0], block_size[1], n_bin) \
               .transpose((1, 0, 2, 3, 4))  # index blocks by rows first
    # hog_feats now contains the gradient amplitudes for each direction,
    # for each cell of its group for each group. Indexing is by rows then columns.

    gradients = np.zeros((n_cells[0], n_cells[1], n_bin))

    # count cells (border cells appear less often across overlapping groups)
    cell_count = np.full((n_cells[0], n_cells[1], 1), 0, dtype=int)

    for off_y in range(block_size[0]):
        for off_x in range(block_size[1]):
            gradients[off_y:n_cells[0] - block_size[0] + off_y + 1,
                  off_x:n_cells[1] - block_size[1] + off_x + 1] += \
                hog_feats[:, :, off_y, off_x, :]
            cell_count[off_y:n_cells[0] - block_size[0] + off_y + 1,
                   off_x:n_cells[1] - block_size[1] + off_x + 1] += 1

    # Average gradients
    gradients /= cell_count

    # Upscale HOG image
    # gradients = cv2.resize(gradients, dsize=(gradients.shape[0] * 4, gradients.shape[1] * 4), interpolation=cv2.INTER_CUBIC)
    gradients = scipy.ndimage.zoom(gradients, cell_size[0], order=0)[:,:,::cell_size[0]]

    return gradients


def pad_patch_HOG(img, pts, h, w, n_bin=9, cell_size=4):
    '''
    Pad cropped HOG discriptor patch
    Args
        img: target HOG descriptor image that will be cropped (h, w, n_bin)
        pts: numpy array of patch coordinates (4, 2)
        h: the padding height (>= than cropped image height)
        w: the padding width (>= than cropped image width)
        n_bin: bin size for HOG descriptor
        cell_size: cell size for HOG descriptor
    Return
        cropped and padded patch
    '''
    # h = int(np.ceil(h / cell_size)) + 1
    # w = int(np.ceil(w / cell_size)) + 1
    # pts = np.ceil(pts / cell_size).astype(np.int32)

    padded = np.zeros((h, w, n_bin), np.float)

    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    cropped = img[y:y+h, x:x+w, :].copy()

    # make mask
    pts = pts - pts.min(axis=0)

    mask = np.zeros(cropped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (1, 1, 1), -1, cv2.LINE_AA)

    mask = np.tile(mask, (9, 1, 1))
    mask = np.transpose(mask, (1, 2, 0))
    # do bit-op
    #  masked = cv2.bitwise_and(cropped, cropped, mask=mask)
    masked = cropped * mask
    padded[0:masked.shape[0], 0:masked.shape[1], :] = masked

    padded = np.reshape(padded, (-1, 1))

    return padded


