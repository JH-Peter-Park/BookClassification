import numpy as np
import cv2

src = cv2.imread('test7.png')
dst = src.copy()
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, 1500, 3000, apertureSize = 5, L2gradient = False)
minLineLength = 500
maxLineGap = 50
lines = cv2.HoughLinesP(canny,1,np.pi/360,150,minLineLength,maxLineGap)
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
    cv2.line(dst, (xMean, 0), (xMean, src.shape[1]), (0, 0, 255), 2)


cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
