import numpy as np
import cv2

src = cv2.imread('img/test9.jpg')
dst = src.copy()
# src_resized = cv2.resize(src, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR) 이미지 크기는 상관없는듯
# blur = cv2.GaussianBlur(src, (5,5), 1)
meanshift = cv2.pyrMeanShiftFiltering(src,10,100,5)
gray = cv2.cvtColor(meanshift, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, 1000, 3000, apertureSize = 5, L2gradient = False)

lines = cv2.HoughLinesP(canny,1,np.pi/180,100,lines = 500,minLineLength = 100,maxLineGap = 15)
extended_lines = []
for line in lines:
    x1,y1,x2,y2 = line[0]
    slope = (y2 - y1) / (x2 - x1)
    if(slope > 2 or slope < -2):
        top_X = int(((0 - y1) / slope) + x1)
        bot_X = int(((src.shape[1] - y1) / slope) + x1)
        extended_lines.append([top_X, bot_X])
        # cv2.line(dst, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.line(dst, (top_X, 0), (bot_X, src.shape[1]), (0, 0, 255), 2)



    # xMean = int((x1+x2)/2)
    # print(line[0])
    # xCoordMeans.append(xMean)
    # print(xMean)

    # height = src.shape[0]
    # topCoord = slope * (0 - x1) + y1
    # bottomCoord = slope * ( height - x1) + y1

    # cv2.line(dst, (xMean, 0), (xMean, src.shape[1]), (0, 0, 255), 2)
cv2.imshow('meanshift', meanshift)
cv2.imshow('canny', canny)
cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
