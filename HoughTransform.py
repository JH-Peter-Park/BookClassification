from typing import List, Any, Union, Tuple

import numpy as np
import cv2
import math

def calcDist(x1,y1,x2,y2):
    return math.sqrt( math.pow(x2-x1,2) + math.pow(y2-y1,2) )

src = cv2.imread("img/test9.jpg")
dst = src.copy()
meanshift = cv2.pyrMeanShiftFiltering(src,15,100)
gray = cv2.cvtColor(meanshift, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, 1000, 3000, apertureSize = 5, L2gradient = False)
# cv2.Canny(이미지, minthres, maxthres, apertureSize, L2gradient)
lines_polar = cv2.HoughLines(canny, 1, np.pi / 180, 150, srn = 0, stn = 0, min_theta = np.radians(170), max_theta = np.radians(190))
# cv2.HoughLines(검출 이미지, 거리, 각도, 임곗값, 거리 약수, 각도 약수, 최소 각도, 최대 각도)

lines_cartesian = []

for i in lines_polar: # 극좌표 -> 직교좌표 변환
    rho, theta = i[0][0], i[0][1]
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a * rho, b * rho

    scale = src.shape[0] + src.shape[1]  # src.shape[1]는 x크기 src.shape[0]이 y크기
    x1 = int(x0 + scale * -b)
    y1 = int(y0 + scale * a)
    x2 = int(x0 - scale * -b)
    y2 = int(y0 - scale * a)

    if x1 == x2: # y축에 평행
        slope = 9999
        lines_cartesian.append(slope, 0, x1)  #y절편은 0으로 설정
    else:
        slope = int((y2 - y1) / (x2 - x1)) #기울기
        intercept_Y = int(y1 - (slope * x1))
        lines_cartesian.append([slope, intercept_Y, 0])

lines_cartesian = sorted(lines_cartesian)  #정렬


for i in lines_cartesian:
    scale = src.shape[0] + src.shape[1]  # src.shape[1]는 x크기 src.shape[0]이 y크기

    slope = i[0]
    intercept_Y = i[1]
    coord_X = i[2]

    y1 = 0
    y2 = src.shape[0]

    if coord_X != 0: # Y축에 평행
        x1 = int(coord_X)
        x2 = int(coord_X)
    else:
        x1 = int((0 - intercept_Y) / slope)
        x2 = int((y2 - intercept_Y) / slope)


    cv2.line(dst, (x1, y1), (x2, y2), (0, 0, 255), 2)


print(lines_cartesian)

cv2.imshow("canny", canny)
cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

# count = 0
#
# rho_curr = 0
# rho_prev = 0
# rho_sum = 0
# rho_mean = 0
#
# theta_curr = 0
# theta_sum = 0
# theta_mean = 0
#
# book_distance = 30

# for i in lines_coord:
#     rho_curr = i[0]
#     theta_curr = i[1]
#
#     if rho_prev != 0:  # 맨 첫번째 직선이 아닌경우
#
#         if abs(rho_prev - rho_curr) < book_distance:  # 직선간 거리 100픽셀 미만
#             rho_sum += rho_curr
#             theta_sum += theta_curr
#             count += 1
#         else:  # 직선간 거리 100픽셀 이상
#             rho_mean = rho_sum / count
#             theta_mean = theta_sum / count
#             lines_mean.append([rho_mean, theta_mean])
#             # 초기화
#             rho_mean = 0
#             rho_sum = 0
#             theta_sum = 0
#             count = 0
#     rho_prev = rho_curr