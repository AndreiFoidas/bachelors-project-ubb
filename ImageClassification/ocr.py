
import cv2
import pytesseract
import os
from PIL import Image
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Tesseract\tesseract.exe'

res_path = "testPhotos"


# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


# erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)


# skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated


# template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

'''
custom_config = r'--oem 3 --psm 6'

image = cv2.imread("testPhotos/test.jpg")
gray = get_grayscale(image)
thresh = thresholding(gray)
opening = opening(gray)
canny = canny(gray)

print("1" + pytesseract.image_to_string(image, config=custom_config))
print("2" + pytesseract.image_to_string(gray, config=custom_config))
print("3" + pytesseract.image_to_string(thresh, config=custom_config))
print("4" + pytesseract.image_to_string(canny, config=custom_config))
print("\n\n")

img = cv2.imread('testPhotos/test.jpg')

h, w, c = img.shape
boxes = pytesseract.image_to_boxes(img, config=custom_config)
for b in boxes.splitlines():
    b = b.split(' ')
    img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
'''

for img in os.listdir(res_path):
    img_name = img
    image_path = res_path + "/" + img
    #img_arr = cv2.imread(image_path)

    custom_config = r'--oem 3 --psm 6'

    print("Testing " + str(image_path))
    img = cv2.imread(image_path)
    gray = get_grayscale(img)
    thresh = thresholding(gray)
    #opening = opening(gray)
    #canny = canny(gray)

    norm_img = np.zeros((img.shape[0], img.shape[1]))
    img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)
    img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]
    img = cv2.GaussianBlur(img, (1, 1), 0)

    #text = pytesseract.image_to_string(img, config=custom_config)
    text = pytesseract.image_to_data(img)
    print(text)

    '''
    print("1 - " + pytesseract.image_to_string(image, config=custom_config))
    print("2 - " + pytesseract.image_to_string(gray, config=custom_config))
    print("3 - " + pytesseract.image_to_string(thresh, config=custom_config))
    #print("4 - " + pytesseract.image_to_string(canny, config=custom_config))
    print("\n\n")
    '''


    image = cv2.imread(image_path)
    '''
    h, w, c = img.shape
    boxes = pytesseract.image_to_boxes(img, config=custom_config)
    for b in boxes.splitlines():
        b = b.split(' ')
        img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

    #cv2.imshow('img', img)
    #cv2.waitKey(0)
    '''
