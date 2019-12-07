import numpy as np
import cv2
my_img = np.zeros((400, 400, 3), "uint8")
pts = np.array([[51,51],[51,100],[100,100],[100,51]])
pts = pts.reshape((-1,1,2))
cv2.fillPoly(my_img,[pts],(0,255,255))
pts1 = np.array([[75,75],[75,125],[125,125],[125,75]], np.int32)
pts1 = pts1.reshape((-1,1,2))
# cv2.fillPoly(my_img,[pts1],(0,0,255))
my_img = cv2.cvtColor(my_img, cv2.COLOR_BGR2GRAY)
area = cv2.countNonZero(my_img)
my_img[my_img==255] = 0

# rotMat = cv2.getRotationMatrix2D((51,51), 45, 1)
# pts = np.array([[51,51],[51,100],[100,100],[100,51]])
# pts = np.concatenate((pts, np.ones((4,1))), 1)
# rotPts = np.matmul(rotMat, pts.T).T
# rotPts = rotPts.astype(int)
# my_img2 = np.zeros((400, 400, 3), "uint8")
# cv2.fillPoly(my_img2,[rotPts],(0,0,255))
# # my_img = cv2.circle(my_img, (51,100), 20, (255,255,255), -1)
# # my_img = cv2.cvtColor(my_img, cv2.COLOR_BGR2GRAY)
print(area)
# my_img = my_img + my_img2
cv2.imshow('Window', my_img)
cv2.waitKey(0)
cv2.destroyAllWindows()