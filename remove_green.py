#!/usr/bin/env python
from typing import Tuple

import cv2
import numpy as np

# cap = cv2.VideoCapture('20210624_nV58_Green_cable.mp4')
cap = cv2.VideoCapture('test_split_body_with_cable.avi')


def make_hsv_range(low: Tuple[int, int, int], up: Tuple[int, int, int]
                   ) -> Tuple[np.ndarray, np.ndarray]:
    return np.array(low), np.array(up)


# green_ = make_hsv_range((0, 0, 0, ), (255, 47, 255,))
green_ = make_hsv_range((0, 0, 0, ), (71, 66, 255,))
# green_ = make_hsv_range((25, 52, 72, ), (200, 255, 255,))
# green_ = make_hsv_range((36, 50, 70, ), (89, 255, 25,))

len_video = cap.get(cv2.CAP_PROP_FRAME_COUNT)
ret, frame = cap.read()

i = 1
# while ret and i < 1000000000:
while ret:
    green_mask = cv2.inRange(frame, *green_)
    # frame[green_mask != 0] = (255, 255, 255,)
    # frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

    # cv2.imshow('img', frame)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        green_mask)
    stats = stats[1:]  # [stats[:,2]/stats[:,3] < 10]
    import pprint
    pprint.pprint(stats)
    print("{}/{}".format(i, len_video))
    # TODO: サイズでソート、a_,b_=sorted([w,h]);a_/b_>0.1のなかでサイズが最大なものを選択
    max_idx = np.argmax(stats[:, -1]) + 1
    for _ in range(1, nlabels):
        if _ != max_idx:
            green_mask[labels == _] = 0

    cv2.imshow('img1', green_mask)
    cv2.imshow('img2', frame)
    # b = np.concatenate((cv2.cvtColor(green_mask, cv2.COLOR_BGR2GRAY),
    #                     frame))
    # cv2.imshow('compare', cv2.resize(b, (250, 300)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ret, frame = cap.read()
    i += 1
else:
    cap.release()
    cv2.destroyAllWindows()
