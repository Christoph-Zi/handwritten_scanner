import easyocr
import matplotlib.pyplot as plt
import numpy as np
import cv2


cam = cv2.VideoCapture(1)
reader = easyocr.Reader(['en'],gpu = False)

while(True):

    #open Webcam
    ret, frame = cam.read()
    cv2.imshow('Webcam', frame)

    #Save Picture
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('data/img.png', frame)
        print('Saved image')

        image = cv2.imread('data/img.png')
        ##sharpened_image = cv2.Laplacian(image, cv2.CV_64F)
        ##sharpened_image = np.uint8(np.absolute(sharpened_image))
        cv2.imshow("Image", image)

        result = reader.readtext(image)
        for detection in result:
            print(detection[1])


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
