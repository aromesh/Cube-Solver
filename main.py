import numpy as np
import math
import cv2

from face import Face

# in RGB 
color_map = {
    'r': ([255, 0, 0]),
    'o': ([255, 165, 0]),
    'b': ([0, 0, 255]),
    'g': ([0, 255, 0]),
    'y': ([255, 255, 0]),
    'w': ([255, 255, 255])
}

# in openCV HSV
color_boundaries = {
    'red': ([0, 75, 75], [10, 255, 255]),
    'red2': ([155, 75, 75], [179, 255, 255]),
    'blue': ([93, 140, 160], [135, 255, 255]),
    'green': ([43, 75, 75], [85, 255, 255]),
    #'orange': ([11, 80, 100], [27, 255, 255]),
}

def draw_cube_rep(frame, face_list):

    d = 50
    start_pos = np.array([100,100])
    for i in range(len(face_list)):
        x = i % 3
        y = math.floor(i/3)
        point_0 = np.array([x, y])*d + 100
        point_1 = np.array([x+1, y+1])*d + 100

        face_color = RGB_to_BGR(face_list[i].get_face_color())

        cv2.rectangle(frame, point_0, point_1, face_color, -1)
        cv2.rectangle(frame, point_0, point_1, (0,0,0), 3)
        text = str(face_list[i].id)
        
        midpoint = ((point_0 + point_1)/2).astype(np.int64)
        cv2.putText(frame, text, midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)


def RGB_to_BGR(color):
    r = color[0]
    g = color[1]
    b = color[2]

    return (b,g,r)

def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return cnts, boundingBoxes

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

    # Min area rectangle contour

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    cv2.drawContours(image,[box],0,(255,0,0),3)


def hsvConverter(color):

    # Function to convert regular HSV values to openCV HSV values

    h = color[0]
    s = color[1]
    v = color[2]

    h_new = round(h/2)
    s_new = round(s/100*255)
    v_new = round(v/100*255)

    return np.array([h_new, s_new, v_new], np.uint8)

def contour_sort(face_list):

    # Function to sort contours from top-to-bottom and left-to-right

    # sorting criteria
    sort_y = lambda elem: elem.y
    sort_x = lambda elem: elem.x

    # Sort by y value
    face_list.sort(key=sort_y)
    
    sorted_face_list = []
    for i in range(len(face_list)):
        
        # Sort each row by x-value
        if (i % 3 == 0):
            row_list = face_list[i:i+3]
            row_list.sort(key=sort_x)
            sorted_face_list[i:i+3] = row_list

    # Re-assign id
    for j in range(len(sorted_face_list)):
        sorted_face_list[j].id = j+1

    return sorted_face_list


### Start script
capture = cv2.VideoCapture(0)

if not (capture.isOpened()):
    print("Could not open camera")

AREA_LIMIT = 600

# Kernel for morphological transforms
kernel_1 = np.ones((5,5),np.uint8)
kernel_2 = np.ones((3,3),np.uint8)

while(True):
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)    
    
    # Convert BGR to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Draw Contours
    face_list = []
    for color_name, (lower, upper) in color_boundaries.items():
        
        # Color bounds
        color_lower = np.array(lower, np.uint8)
        color_upper = np.array(upper, np.uint8)

        # Color mask and transformation
        color_mask = cv2.inRange(hsv_frame, color_lower, color_upper)
        opening = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel_1, iterations=5)
        gradient = cv2.morphologyEx(opening, cv2.MORPH_GRADIENT, kernel_2)    

        contours, hierarchy = cv2.findContours(gradient,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        # Loop through contours based on area and draw them
        contour_id = 0
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if (contour_area > AREA_LIMIT):

                # Find centre of contour
                M = cv2.moments(contour)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])

                drawContour(contour, frame)
                cv2.circle(frame,(cx,cy), 2, (0,255,0), -1)

                contour_id += 1
                face_instance = Face(color_name[0], cx, cy, color_map, contour_id)
                face_list.append(face_instance)

    # sort contours
    face_list = contour_sort(face_list)

    # Draw ID on actual cube
    for i in range(len(face_list)):
        
        text = str(face_list[i].id)
        face = face_list[i]
        cv2.putText(frame, text, (face.x,face.y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

    # Draw captured cube in window
    draw_cube_rep(frame, face_list)
    
    # reset face list
    face_list.clear()

    cv2.imshow("Main Window", frame)
    cv2.imshow("Opening", opening)
    cv2.imshow("Gradient", gradient)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
capture.release()
