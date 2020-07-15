import numpy as np
import cv2
import math

def calcDist(x1,y1,x2,y2):
    return math.sqrt( math.pow(x2-x1,2) + math.pow(y2-y1,2) )

src = cv2.imread("img/test9.jpg")
dst = src.copy()
meanshift = cv2.pyrMeanShiftFiltering(src,10,100,5)
gray = cv2.cvtColor(meanshift, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, 1000, 2000, apertureSize = 5, L2gradient = False) #1500 1000
# cv2.Canny(이미지, minthres, maxthres, apertureSize, L2gradient)
lines = cv2.HoughLines(canny, 0.6, np.pi / 180, 150, srn = 0, stn = 0, min_theta = np.radians(170), max_theta = np.radians(190))
# cv2.HoughLines(검출 이미지, 거리, 각도, 임곗값, 거리 약수, 각도 약수, 최소 각도, 최대 각도)
lines_coord = []

for i in lines:
    rho, theta = i[0][0], i[0][1]
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a*rho, b*rho

    scale = src.shape[0] + src.shape[1]
    x1 = int(x0 + scale * -b)
    y1 = int(y0 + scale * a)
    x2 = int(x0 - scale * -b)
    y2 = int(y0 - scale * a)
    lines_coord.append([(x1,y1),(x2,y2)])
    cv2.line(dst, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.circle(dst, (x0, y0), 3, (255, 0, 0), 5, cv2.FILLED)
print(lines_coord)
cv2.imshow('meanshift', meanshift)
cv2.imshow("canny", canny)
cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

