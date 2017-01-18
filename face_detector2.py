#! /usr/bin/python
# -*- coding: utf-8 -*-
u"""dlibによる顔画像検出."""
import cv2
import dlib
import numpy as np

#入力はx*y*3の画像 (openCVのimreadで読み込んだやつ)。出力はz*112*112*3の画像の配列（顔が複数画像内にあった場合はzは複数になる）
def getFaces(input_image):
    detector = dlib.get_frontal_face_detector()
    results = []
    leftbottoms = []
    height, width = input_image.shape[:2]
    rects, scores, types = detector.run(input_image,0)
    for i, rect in enumerate(rects):
        top, bottom, left, right = rect.top(), rect.bottom(), rect.left(), rect.right()
        if min(top, height - bottom - 1, left, width - right - 1) < 0:
            continue
        results.append(cv2.resize(input_image[top : bottom, left : right],(112,112)))
        leftbottoms.append((left, bottom))
        cv2.rectangle(input_image, (left, top), (right, bottom), (0, 255, 0), 2)
    return results, leftbottoms

if __name__ == '__main__':
    image = cv2.imread("beatles.jpg")
    output = getFaces(image)
    print(np.array(output[0]).shape)
    cv2.imshow("test",output[3])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
