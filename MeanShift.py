import numpy as np
import cv2

src = cv2.imread('img/test1.jpg')
dst = src.copy()
# blur = cv2.bilateralFilter(src,9,100,100)
blur = cv2.GaussianBlur(src, (5,5), 0)
meanshift = cv2.pyrMeanShiftFiltering(blur,10,50,2)
canny = cv2.Canny(meanshift, 1500, 3000, apertureSize = 5, L2gradient = False) #1500 1000
lines = cv2.HoughLines(canny, 0.6, np.pi / 180, 150, srn = 0, stn = 0, min_theta = np.radians(170), max_theta = np.radians(190))
# cv2.HoughLines(검출 이미지, 거리, 각도, 임곗값, 거리 약수, 각도 약수, 최소 각도, 최대 각도)


for i in lines:
    rho, theta = i[0][0], i[0][1]
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a*rho, b*rho

    scale = src.shape[0] + src.shape[1]

    x1 = int(x0 + scale * -b)
    y1 = int(y0 + scale * a)
    x2 = int(x0 - scale * -b)
    y2 = int(y0 - scale * a)

    cv2.line(dst, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.circle(dst, (x0, y0), 3, (255, 0, 0), 5, cv2.FILLED)

cv2.imshow("canny", canny)
cv2.imshow("dst", dst)

cv2.waitKey(0)
cv2.destroyAllWindows()