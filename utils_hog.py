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
    match_groups = []
    for group1, group2 in zip(patch_groups1, patch_groups2):
        if len(group1) == 0:
            continue
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

    return gradients


