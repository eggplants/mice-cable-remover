#!/usr/bin/env python
from glob import glob
from typing import Tuple

import cv2
import numpy as np

datas = glob('data/*') + glob('../data_with_cable/*')

for i, n in enumerate(datas):
    print(i, ":", n)

s = input("enter number: ")
cap = cv2.VideoCapture(datas[int(s)])


def make_hsv_range(low: Tuple[int, int, int], up: Tuple[int, int, int]
                   ) -> Tuple[np.ndarray, np.ndarray]:
    return np.array(low), np.array(up)


def resize_frame(frame: np.ndarray, ratio: float = 0.5) -> np.ndarray:
    h, w = frame.shape[0:2]
    return cv2.resize(frame, (int(w*ratio), int(h*ratio)))


#  = make_hsv_range((0, 0, 0, ), (255, 47, 255,))
#  = make_hsv_range((0, 0, 0, ), (71, 66, 255,))
#  = make_hsv_range((0, 0, 0, ), (71, 34, 23,))
filter_ = make_hsv_range((0, 0, 0, ), (45, 255, 23,))
#  = make_hsv_range((25, 52, 72, ), (200, 255, 255,))
#  = make_hsv_range((36, 50, 70, ), (89, 255, 25,))

len_video = cap.get(cv2.CAP_PROP_FRAME_COUNT)
ret, frame = cap.read()

i = 1
x, y = 0, 0
while ret:
    mask = cv2.inRange(frame, *filter_)
    # mask = cv2.dilate(mask, kernel1)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel1)
    # mask = cv2.erode(mask, kernel2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((20, 20), np.uint8))
    mask = cv2.dilate(mask, np.ones((50, 50), np.uint8))

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, 4)
    stats = stats[1:]
    for _ in range(len(stats)):
        a_, b_ = np.sort(stats[_, 2:4])
        if a_/b_ < 0.1:
            stats[_, -1] = 0
    if len(stats) > 0:
        max_ind = np.argmax(stats[:, -1]) + 1
        x, y = centroids[max_ind]
    else:
        input()
        pass
    binarized_frame = cv2.circle(mask,
                                 (int(x), int(y), ),
                                 10, (150, 150, 150),  thickness=4)
    mono = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    side_by_side = np.hstack([mono, frame])
    cv2.imshow('a', resize_frame(side_by_side))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    #################################
    print("[{}/{}]".format(i, len_video))
    print("biggest components center:", "({}, {})".format(x, y))
    print("components:", len(stats))
    #################################

    ret, frame = cap.read()
    i += 1
else:
    cap.release()
    cv2.destroyAllWindows()
