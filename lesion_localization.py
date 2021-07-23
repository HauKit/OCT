import cv2
import numpy as np


def lesion_localization(image, visual_cam, cls, possi):
    local_cam = cv2.resize(visual_cam, dsize=(512, 512))
    st = cls + ' : %.3f' % possi
    cnt = image.copy()
    if cls != 'normal':
        ret, bin = cv2.threshold((local_cam * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #lower = np.array([0])
        #upper = np.array([128])
        #bin = cv2.inRange((local_cam * 255).astype(np.uint8), 0, 200)
        #cv2.imshow('a', bin)
        #bin = cv2.bitwise_not(bin)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        bin = cv2.morphologyEx(bin, 2, kernel)
        bin = cv2.morphologyEx(bin, 3, kernel)
        contours, hierarchy = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(cnt, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(cnt, st, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(cnt, st, (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2)
    return cnt