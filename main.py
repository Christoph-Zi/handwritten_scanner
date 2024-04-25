
import easyocr
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import pytesseract
from PIL import Image



pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def detect_and_draw_rectangles(frame):
    # Konvertiere das Bild in Graustufen
    sharpened = cv2.filter2D(cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR), -1, kernel=np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
    cv2.imshow("Gray Picture", sharpened)

    # Wende den Kantendetektor an
    edges = cv2.Canny(sharpened, 0, 255)

    # Finde Konturen im Bild
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iteriere durch die Konturen und zeichne grüne Rechtecke um Vierecke
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4 and cv2.contourArea(contour) > 20:
            cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)

    return frame




cam = cv2.VideoCapture(1)

while(True):

    #open Webcam
    ret, frame = cam.read()

    frame_with_rectangles = detect_and_draw_rectangles(frame)
    cv2.imshow('Webcam', frame)

    #Save Picture
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('data/img.png', frame)
        print('Saved image')

        img = cv2.imread('data/img.png')
        cv2.imshow("original image", img)

        result = pytesseract.image_to_boxes(img)
        print(result)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

