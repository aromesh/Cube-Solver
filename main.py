import numpy as np
import math
import cv2

AREA_LIMIT = 600

# in RGB 
color_map = {
    'r': ([255, 0, 0]),
    'b': ([0, 0, 255])
}

# in openCV HSV
color_boundaries = {
    'red': ([0, 100, 100], [10, 255, 255]),
    'red2': ([155, 100, 100], [179, 255, 255]),
    'blue': ([93, 140, 160], [135, 255, 255]),
    #'orange': ([11, 110, 125], [27, 255, 255]) 
}

capture = cv2.VideoCapture(0)

if not (capture.isOpened()):
    print("Could not open camera")

def draw_cube_rep(frame, cubie_face):

    d = 50
    start_pos = np.array([100,100])
    pos_1 = start_pos + np.array([-3*d/2, -3*d/2])
    pos_2 = start_pos + np.array([-1*d/2, -1*d/2])
    new_color = cubie_face.face_color()
    cv2.rectangle(frame, pos_1.astype(int), pos_2.astype(int), new_color, -1)

class Face(object):
    def __init__(self, color_name, x, y):
        self.color = color_name
        self.x = x
        self.y = y

    def face_color(self):

        return color_map[self.color]


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

def drawContour(contour, image, contour_color):

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


#morphological transformations
kernel_1 = np.ones((5,5),np.uint8)
kernel_2 = np.ones((3,3),np.uint8)


# take second element for sort
def take_x(elem):
    return elem.x

while(True):
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)    
    
    # Convert BGR to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Draw Contours
    out_number = 0
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
        #print(len(contours))

        if (len(contours) > 0):
            #sort contours top to bottom
            (contours,_) = sort_contours(contours, method="left-to-right")
        
        #print(len(contours))

        cube_rows = []
        row = []
        for (i, c) in enumerate(contours, start=1):
            row.append(c)
            if i % 3 == 0:
                (contours,_) = sort_contours(contours, method="top-to-bottom")
                cube_rows.append(contours)
                row = []


        # Loop through contours based on area an draw them
        number = 0
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if (contour_area > AREA_LIMIT):

                M = cv2.moments(contour)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])

                contour_color = np.mean(np.array([color_lower, color_upper]))
                drawContour(contour, frame, contour_color)
                cv2.circle(frame,(cx,cy), 2, (0,255,0), -1)
                number+=1
                text = str(number)
                #if (len(contours) == 9): 
                cv2.putText(frame, text, (cx,cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                out_number += 1
                face_instance = Face(color_name[0], cx, cy)
                face_list.append(face_instance)


    print(len(face_list))

    # sort stuff
    face_list.sort(key=take_x)

    # loop for drawing faces in window
    for i in range(len(face_list)-1):
        draw_cube_rep(frame, face_list[i])
    

    face_list.clear()
    out_number = 0

    cv2.imshow("Main Window", frame)
    cv2.imshow("Opening", opening)
    cv2.imshow("Gradient", gradient)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
capture.release()
