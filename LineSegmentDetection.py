import cv2
import math
import numpy as np

#Read gray image
img = cv2.imread("test1.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#Create default parametrization LSD
lsd = cv2.createLineSegmentDetector(0)

#Detect lines in the image
lines = lsd.detect(gray)[0] #Position 0 of the returned tuple are the detected lines
print(lines[0][0][0])

ver_lines = []

for line in lines:
    angletan = math.degrees(math.atan2((round(line[0][3],2) - round(line[0][1],2)), (round(line[0][2],2) - round(line[0][0],2))))

    if(angletan > 85 and angletan < 95):
        ver_lines.append(line)

#Draw detected lines in the image
drawn_img = lsd.drawSegments(img,np.array(ver_lines))

#Show image
cv2.imshow("LSD",drawn_img )
cv2.waitKey(0)