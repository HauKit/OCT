import cv2
import numpy as np


def image_feature_extract(frame, blur_flag, feature_enhance_flag):
    blur = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if blur_flag:
        blur = cv2.GaussianBlur(blur, (3, 3), 0)

    lower = np.array([0, 0, 0])
    upper = np.array([254, 254, 254])
    mask = cv2.inRange(frame, lower, upper)
    cnt = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        epsilon = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        cv2.drawContours(cnt, [approx], -1, (255, 255, 255), -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cnt = cv2.erode(cnt, kernel, iterations=1)
    result = cv2.bitwise_and(blur, blur, mask=cnt)

    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel2)
    result = cv2.medianBlur(result, 3)

    ret, otsu = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = np.zeros_like(cnt)
    for c in contours:
        cv2.drawContours(cnt, [c], -1, (255, 255, 255), -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 100))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cnt = cv2.morphologyEx(cnt, 2, kernel2)
    cnt = cv2.morphologyEx(cnt, 3, kernel)
    cnt = cv2.bitwise_and(blur, blur, mask=cnt)

    if feature_enhance_flag:
        cnt = cv2.equalizeHist(cnt)
        return cv2.cvtColor(cnt, cv2.COLOR_GRAY2RGB)
    else:
        return cv2.cvtColor(cnt, cv2.COLOR_GRAY2RGB)


if __name__ == '__main__':
    #frame = cv2.imread('F:/OCT2017/train/CNV/CNV-7565927-212.jpeg')
    frame = cv2.imread('F:/1.png')
    #frame = cv2.imread('F:/OCT2017_Enhance/test/CNV/CNV-154835-1 (2).jpeg')
    cv2.imshow('source', frame)
    test = image_feature_extract(frame, False, False)
    cv2.imshow('test1', test)
    test = image_feature_extract(frame, False, True)
    cv2.imshow('test2', test)
    cv2.waitKey(0)