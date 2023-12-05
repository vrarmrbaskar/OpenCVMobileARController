#import CV - library for Computer vision
import cvzone
from cvzone.ColorModule import ColorFinder 
import cv2
import socket

# setting up webcam.
# '0' specifies single webcam , for multiple cameras, it can be 1,2.
webcam = cv2.VideoCapture(0)
# webcam width / height parameters
webcam.set(3,1920)
webcam.set(4,1080)

#h, w, _ = img.shape
# TRUE to show the track bars to adjust the color
myColorFinder = ColorFinder(False)

#Color value for marker pasted on the CUBE
# This will be used for ColorFinder to get the contours 
# 'hmax', 'smax', 'vmax' are the maximum values for Hue, Saturation, and Value.
hsvVals = {'hmin': 95, 'smin': 7, 'vmin': 39, 'hmax': 99, 'smax': 164, 'vmax': 250}

#Open the socket with port and ipaddress 
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5053)

#opening webcam
while True:
    success,img = webcam.read()
    h, w, _ = img.shape
    imgColor,mask = myColorFinder.update(img,hsvVals)
    imgContour,contours = cvzone.findContours(img,mask,minArea=1000)
    
    #contours[0][0] - X axis
    #contours[0][1] - Y axis
    #contours[0]['area'] - Z axis
    if contours:
        data = contours[0]['center'][0], \
               h - contours[0]['center'][1], \
               int(contours[0]['area'])
        print(data)
        sock.sendto(str.encode(str(data)), serverAddressPort)
    
  
    imgContour = cv2.resize(imgContour, (0, 0), None, 0.5, 0.5)
    cv2.imshow("ImageContour", imgContour)
    cv2.waitKey(1)

