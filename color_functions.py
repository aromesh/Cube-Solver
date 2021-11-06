import colorsys
import numpy as np

def hsvConverter(color):

    # Function to convert regular HSV values to openCV HSV values

    h = color[0]
    s = color[1]
    v = color[2]

    h_new = round(h/2)
    s_new = round(s/100*255)
    v_new = round(v/100*255)

    return np.array([h_new, s_new, v_new], np.uint8)


def BGR_to_HSV(color):
    b = color[0]/255
    g = color[1]/255
    r = color[2]/255

    color = colorsys.rgb_to_hsv(r, g, b)
    color = (int(color[0]*179), int(color[1]*255), int(color[2]*255))
    
    return color

def RGB_to_BGR(color):
    r = color[0]
    g = color[1]
    b = color[2]

    return (b,g,r)