import cv2
import numpy as np
import math
import pickle
import os

PI = math.pi

def calcIoU(img, patch1, patch2):
    patch1 = patch1.reshape((-1,1,2))
    patch2 = patch2.reshape((-1,1,2))

    canvas1 = np.zeros(img.shape, "uint8")
    cv2.fillPoly(canvas1,[patch1],(255,255,255))
    canvas1_grey = cv2.cvtColor(canvas1, cv2.COLOR_BGR2GRAY)
    area1 = cv2.countNonZero(canvas1_grey)

    canvas2 = np.zeros(img.shape, "uint8")
    cv2.fillPoly(canvas2,[patch2],(255,255,255))
    canvas2_grey = cv2.cvtColor(canvas2, cv2.COLOR_BGR2GRAY)
    area2 = cv2.countNonZero(canvas2_grey)

    cv2.fillPoly(canvas1,[patch2],(255,255,255))
    canvas3_grey = cv2.cvtColor(canvas1, cv2.COLOR_BGR2GRAY)
    total = cv2.countNonZero(canvas3_grey)

    IoU = area1 + area2 - total / total

    return IoU

def getRadialPts(center, numPts):
    radialPts = np.ones((numPts + 1, 3))
    for step in range(0,numPts+1):
        X = center[0] + math.cos(2 * PI / numPts * step) * 100
        Y = center[1] + math.sin(2 * PI / numPts * step) * 100
        radialPts[step, :2] = np.array([X, Y])
    return radialPts

def getRadialLines(numLines, center, F):
    radialPts = getRadialPts(center, numLines)
    lines1 = np.cross(center, radialPts)

    radialPts = radialPts[:, :2]
    lines2 = cv2.computeCorrespondEpilines(radialPts.reshape(-1,1,2), 1, F)
    lines2 = lines2.reshape(-1, 3)

    return lines1, lines2, radialPts

def calcDelta(angle, height, overlap=0.66):
    ratio = (1 - overlap) / (1 + overlap)
    deltaLen = height * ratio
    deltaAng = angle * ratio
    return deltaLen, deltaAng

def find_valid_lines(lines, img):
    lines = lines.T
    r,c = img.shape
    bound_down = np.array([0, -1, r])
    bound_up = np.array([0, 1, 0])
    bound_left = np.array([1, 0, 0])
    bound_right = np.array([-1, 0, c])
    bounds = np.array([bound_up, bound_down, bound_left, bound_right])

    intersect_up = np.cross(lines.T, bound_up)
    intersect_up /= np.repeat(intersect_up[:, 2], 3).reshape(-1, 3)
    intersect_down = np.cross(lines.T, bound_down)
    intersect_down /= np.repeat(intersect_down[:, 2], 3).reshape(-1, 3)
    intersect_left = np.cross(lines.T, bound_left)
    intersect_left /= np.repeat(intersect_left[:, 2], 3).reshape(-1, 3)
    intersect_right = np.cross(lines.T, bound_right)
    intersect_right /= np.repeat(intersect_right[:, 2], 3).reshape(-1, 3)
    
    valid_up = (intersect_up[:, 0] >= 0) & (intersect_up[:, 0] < c)
    valid_down = (intersect_down[:, 0] >= 0) & (intersect_down[:, 0] < c)
    valid_left = (intersect_left[:, 1] >= 0) & (intersect_left[:, 1] < r)
    valid_right = (intersect_right[:, 1] >= 0) & (intersect_right[:, 1] < r)

    valid_mask = valid_up | valid_down | valid_left | valid_right
    valid_lines = lines.T[valid_mask, :]
    
    return valid_lines

def getPoints(img, center, lines, delta, overlap=0.66):
    # TODO
    # lines = lines.T
    h, w = img.shape[:2]
    if len(img.shape) == 2:
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
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
        points_on_line = []
        while (True):
            color = tuple(np.random.randint(0,255,3).tolist())

            deltaXY = inc * step
            
            if (delta * step > maxDist):
                break

            point = center[:2] + deltaXY
            points_on_line.append(point)

            ptX, ptY = map(int, point)
            img = cv2.circle(img, (ptX,ptY), 20, color, -1)

            step += 1
        points_all.append(points_on_line)
    return img, points_all

def isInImage(img, patch):
    h, w = img.shape[:2]
    minX = np.min(patch[:,0])
    maxX = np.max(patch[:,0])
    minY = np.min(patch[:,1])
    maxY = np.max(patch[:,1])

    if minX < 0 or minY < 0 or maxX > w or maxY > h:
        return False
    return True

def getPatches(cfg, img, points):
    # points : M x N_m x 2 list for N_m points along the Mth line
    overlap = cfg.OVERLAP
    patch_height = int((1 + overlap) / (1 - overlap))
    patch_angle = int((1 + overlap) / (1 - overlap))
    
    patches = []
    numOfLines = len(points)
    for i in range(numOfLines):
        pointsAlong_top = points[i]
        pointsAlong_bot = points[(i + patch_angle) % numOfLines]
        for j in range(len(pointsAlong_top)):
            pt1 = pointsAlong_top[j]
            pt2 = pointsAlong_top[(j + patch_height) % len(pointsAlong_top)]
            pt3 = pointsAlong_bot[(j + patch_height) % len(pointsAlong_bot)]
            pt4 = pointsAlong_bot[j]

            patch = np.array([pt1, pt2, pt3, pt4])
            if isInImage(img, patch):
                patches.append(patch)
    
    with open(os.path.join(cfg.SAVE_PATH, "patches.pkl"), 'wb') as output:
        pickle.dump(patches, output, -1)

    return patches