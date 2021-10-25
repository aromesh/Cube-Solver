import numpy as np
import math
import cv2

color_boundaries = {
    "red": ([0, 0, 255], [127, 0, 255]),
    "blue": ([90, 50, 50], [130, 255, 255])
}

capture = cv2.VideoCapture(0)

if not (capture.isOpened()):
    print("Could not open camera")

def drawContour(contour, image):

    # Draw rectangular contour with the minimum area


    ## Contour Approximation
    # epsilon = 0.1*cv2.arcLength(contour,True)
    # approx = cv2.approxPolyDP(contour,epsilon,True)

    ## bounding rect
    #x,y,w,h = cv2.boundingRect(contour)
    #cv2.rectangle(frame, (x, y), (x + w, y + h), (36,255,12), 1)

    # Regular contours
    # cv2.drawContours(frame, [contour], 0, (255, 255, 0), 2)

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(image,[box],0,(100, 255, 255),3)


def hsvConverter(color):

    # Function to convert regular HSV values to openCV HSV values

    h = color[0]
    s = color[1]
    v = color[2]

    h_new = math.round(h/2)
    s_new = math.round(s/100*255)
    v_new = math.round(v/100*255)

    return np.array([h_new, s_new, v_new], np.uint8)

    

while(True):
    ret, frame = capture.read()    
    
    # Convert BGR to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Blue color ranges
    blue_lower = np.array([93, 160, 180], np.uint8)
    blue_upper = np.array([135, 255, 255], np.uint8)
    
    # Threshold the HSV image to get only blue colors
    blue_mask = cv2.inRange(hsv_frame, blue_lower, blue_upper)

    # Draw Contours
    contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #morphological transformations
    kernel_1 = np.ones((5,5),np.uint8)
    kernel_2 = np.ones((3,3),np.uint8)

    opening = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel_1, iterations=5)
    gradient = cv2.morphologyEx(opening, cv2.MORPH_GRADIENT, kernel_2)
    
    contours, hierarchy = cv2.findContours(gradient,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    i = 0
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if (contour_area > 500.0):

            M = cv2.moments(contour)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            drawContour(contour, frame)

            i +=1
    print(i)

    cv2.imshow("Main Window", frame)
    cv2.imshow('Blue Mask', blue_mask)
    cv2.imshow("Opening", opening)
    cv2.imshow("Gradient", gradient)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
capture.release()
