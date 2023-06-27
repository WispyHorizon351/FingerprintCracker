import cv2
import math
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

fingerprints = {
    'print1': cv2.imread('assets/prints/fprint1.png'),
    'print2': cv2.imread('assets/prints/fprint2.png'),
    'print3': cv2.imread('assets/prints/fprint3.png'),
    'print4': cv2.imread('assets/prints/fprint4.png'),
    'print5': cv2.imread('assets/TestCases/testimage.png', 0)
}

combos = {
    'print1': ['1', '2', '3', '4'],
    'print2': ['5', '6', '7', '8'],
    'print3': ['9', '10', '11', '12'],
    'print4': ['13', '14', '15', '16']
}


for i in range(1, 5):
    title = f'print{i}'
    template = fingerprints[title]

    file = f'assets/TestCases/{title}img.png'

    img_rgb = cv2.imread(file)

    h, w, c = img_rgb.shape
    nW = math.floor(w * (2/3))
    nH = math.floor(h * (2/3))

    img_rgb = cv2.resize(img_rgb, (nW, nH))

    def findImage(image, template, method):
        h, w, c = template.shape
        res = cv2.matchTemplate(image, template, method)
        threshold = 0.8
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    findImage(img_rgb, template, cv2.TM_CCOEFF_NORMED)
    for i in combos[title]:
        tempimg = cv2.imread(f'assets/fprintParts/piece{i}.png')
        findImage(img_rgb, tempimg, cv2.TM_CCOEFF_NORMED)

    cv2.imshow('Match', img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    