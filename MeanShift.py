import numpy as np
import cv2
# meanshift 문제점: 같은색깔끼리 edge blur처리해버림
src = cv2.imread('img/test1.jpg')
dst = src.copy()
# blur = cv2.bilateralFilter(src,9,100,100)
# blur = cv2.GaussianBlur(src, (5,5), 1)
meanshift = cv2.pyrMeanShiftFiltering(src,10,100,3)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, 1500, 3000, apertureSize = 5, L2gradient = False)
minLineLength = 300
maxLineGap = 50
lines = cv2.HoughLinesP(canny,1,np.pi/360,10,minLineLength,maxLineGap)
xCoordMeans = []
for line in lines:
    x1,y1,x2,y2 = line[0]
    if(y1 == y2):
        continue
    xMean = int((x1+x2)/2)
    print(line[0])
    xCoordMeans.append(xMean)
    print(xMean)
    # slope = (y2-y1)/(x2-x1)
    # height = src.shape[0]
    # topCoord = slope * (0 - x1) + y1
    # bottomCoord = slope * ( height - x1) + y1
    cv2.line(dst,(x2,y2),(x1,y1),(0,255,0),2)
    # cv2.line(dst, (xMean, 0), (xMean, src.shape[1]), (0, 0, 255), 2)

cv2.imshow("meanshift", meanshift)
cv2.imshow("canny", canny)
cv2.imshow("dst", dst)

cv2.waitKey(0)
cv2.destroyAllWindows()