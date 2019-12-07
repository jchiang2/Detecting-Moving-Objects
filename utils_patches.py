import cv2
import numpy as np
import math
import pickle
import os

PI = math.pi

def calcArea(img, patches):
    '''
    Calculates the area of a patch.
    Args:
        img: (numpy array h x w x 3) Origin image
        patches: (list N x 4 x 2) List of patches
    returns:
        area: (int) Total area of the patches
    '''
    canvas = np.zeros(img.shape, "uint8")
    for p in patches:
        p = p.reshape((-1,1,2))
        p = p.astype(int)
        cv2.fillPoly(canvas,[p],(255,255,255))
    if len(canvas.shape) == 3:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    area = cv2.countNonZero(canvas)
    return area

def calcIoU(img, patch1, patch2):
    '''
    Calculates the IoU of two patches.
    Args:
        img: (numpy array h x w x 3) Origin image
        patch1(patch2): (numpy array 4 x 2) A patch represented by its four vertices
    returns:
        IoU: (float) The intersection-over-union
    '''
    patch1 = patch1.reshape((-1,1,2))
    patch2 = patch2.reshape((-1,1,2))

    area1 = calcArea(img, [patch1])
    area2 = calcArea(img, [patch2])
    total = calcArea(img, [patch1, patch2])

    IoU = (area1 + area2 - total) / total

    return IoU

def getRadialPts(center, numPts):
    '''
    Returns a set of equally spaced radial points from a center point.
    Args:
        center: (numpy array 1 x 3) Coordinates of the center point
        numPts: (int) Number of desired circular points
    returns:
        radialPts: (numpy array (numPts + 1) x 3) Array of homogenous coordinates of radial points
    '''
    radialPts = np.ones((numPts + 1, 3))
    for step in range(0,numPts+1):
        X = center[0] + math.cos(2 * PI / numPts * step) * 100
        Y = center[1] + math.sin(2 * PI / numPts * step) * 100
        radialPts[step, :2] = np.array([X, Y])
    return radialPts

def getRadialLines(numLines, center):
    '''
    Returns a set of equally spaced coincidental lines.
    Args:
        numLines: (int) Number of desired radial lines
        center: (numpy array 1 x 3) Homogenous coordinates of the center point
    returns:
        lines: (numpy array N x 3) List of N equally spaced, coincidental lines
        radialPts: (numpy array (numLines + 1) x 2) Array of 2D radial points
    '''
    radialPts = getRadialPts(center, numLines)
    lines = np.cross(center, radialPts)
    radialPts = radialPts[:, :2]
    return lines, radialPts

def find_valid_lines(img, lines, pts):
    '''
    Returns lines visible in the image.
    Args:
        img: (numpy array h x w) Reference image (grayscale)
        lines: (numpy array N x 3) Set of N epipolar lines
        pts: (numpy array N x 2) Set of N corresponding points on the lines
    returns:
        valid_lines: (numpy array M x 3) Set of M valid epipolar lines
        valid_pts: (numpy array M x 2) Set of M valid points
    '''
    r,c = img.shape
    bound_down = np.array([0, -1, r])
    bound_up = np.array([0, 1, 0])
    bound_left = np.array([1, 0, 0])
    bound_right = np.array([-1, 0, c])
    bounds = np.array([bound_up, bound_down, bound_left, bound_right])

    intersect_up = np.cross(lines, bound_up)
    intersect_up /= np.repeat(intersect_up[:, 2], 3).reshape(-1, 3)
    intersect_down = np.cross(lines, bound_down)
    intersect_down /= np.repeat(intersect_down[:, 2], 3).reshape(-1, 3)
    intersect_left = np.cross(lines, bound_left)
    intersect_left /= np.repeat(intersect_left[:, 2], 3).reshape(-1, 3)
    intersect_right = np.cross(lines, bound_right)
    intersect_right /= np.repeat(intersect_right[:, 2], 3).reshape(-1, 3)
    
    valid_up = (intersect_up[:, 0] >= 0) & (intersect_up[:, 0] < c)
    valid_down = (intersect_down[:, 0] >= 0) & (intersect_down[:, 0] < c)
    valid_left = (intersect_left[:, 1] >= 0) & (intersect_left[:, 1] < r)
    valid_right = (intersect_right[:, 1] >= 0) & (intersect_right[:, 1] < r)

    valid_mask = valid_up | valid_down | valid_left | valid_right
    valid_lines = lines[valid_mask, :]
    valid_pts = pts[valid_mask, :]
    
    return valid_lines, valid_pts

def find_corr_lines(radialPts, F):
    '''
    Finds correspoinding epipolar lines in a support image.
    Args:
        radialPts: radialPts: (numpy array (numPts + 1) x 2) Array of 2D radial points
        F: (numpy array 3 x 3) Fundamental matrix
    returns:
        lines: (numpy array N x 3) Corresponding epipolar lines in the 2D image
    '''
    lines = cv2.computeCorrespondEpilines(radialPts.reshape(-1,1,2), 1, F)
    lines = lines.reshape(-1, 3)
    return lines

def getVertices(img, center, lines, delta):
    '''
    Returns the set of all possible patch vertices in an image.
    Args:
        img: (numpy array h x w x 3) Origin image
        center: (numpy array 1 x 3) Homogenous coordinates of the center point
        lines: (numpy array N x 3) Lines in the 2D image
        delta: (float) Radial distance between points
    returns:
        points_all: (list N x M_n x 2) Set of M_n points along Nth line
    '''
    h, w = img.shape[:2]
    center = np.copy(center)
    vertices = np.array([[0,0],[w,0],[w,h],[0,h]])
    diff = vertices - center[:2]
    dist = [np.linalg.norm(diff[i]) for i in range(diff.shape[0])]
    maxDist = max(dist)
    
    angles = np.arctan2(-lines[:,0], lines[:,1])
    deltaX = np.cos(angles) * delta
    deltaY = np.sin(angles) * delta
    
    points_all = []
    for i in range(deltaX.shape[0]):
        inc = np.array([deltaX[i], deltaY[i]])
        step = 0
        count = -1
        points_on_line = []
        while (True):
            deltaXY = inc * step

            if count == step:
                break
            if (delta * step > maxDist):
                count = step
                step = 0
                inc = -inc

            point = center[:2] + deltaXY
            points_on_line.append(point)
            step += 1
        points_all.append(points_on_line)
    return points_all

def isInImage(img, patch):
    '''
    Checks if a patch is within an image.
    Args:
        img: (numpy array h x w x 3) Origin image 
        patch: (numpy array 4 x 2) A patch represented by its four vertices
    returns:
        bool: True if patch is in image
    '''
    h, w = img.shape[:2]
    minX, minY = np.amin(patch, axis=0)
    maxX, maxY = np.amax(patch, axis=0)

    if minX < 0 or minY < 0 or maxX > w or maxY > h:
        return False
    return True

def getPatches(cfg, img, points, isSupport=False):
    '''
    Samples vertices into patches.
    Args:
        cfg: Config object
        img: (numpy array h x w x 3) Origin image
        points: (list N x M_n x 2) Set of M_n points along Nth line
        isSupport: (bool) True if using support image (sample different widths)
    returns:
        patch_groups: (list N x M x 4 x 2) Patch groups for N pairs of epipolar lines, 
                      each with M patches
    '''
    overlap = cfg.OVERLAP
    patch_height = int(np.ceil((1 + overlap) / (1 - overlap)))
    patch_angle = int(np.ceil((1 + overlap) / (1 - overlap)))
    
    patch_groups = []
    numOfLines = len(points)
    for i in range(numOfLines):
        pointsAlong_top = points[i]
        pointsAlong_bot = points[(i + patch_angle) % numOfLines]
        patches = []
        for j in range(len(pointsAlong_top)):
            if isSupport:
                # patch_var = [-1, 0, 1]
                patch_var = [0]
            else:
                patch_var = [0]
            for var in patch_var:
                pt1 = pointsAlong_top[j]
                pt2 = pointsAlong_top[(j + patch_height + var) % len(pointsAlong_top)]
                pt3 = pointsAlong_bot[(j + patch_height + var) % len(pointsAlong_bot)]
                pt4 = pointsAlong_bot[j]

                patch = np.array([pt1, pt2, pt3, pt4])
                if isInImage(img, patch):
                    patches.append(patch)
        patch_groups.append(patches)

    return patch_groups

