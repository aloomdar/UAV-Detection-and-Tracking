import cv2
from filterpy.kalman import KalmanFilter
import numpy as np
import os

directory = os.path.dirname(os.path.abspath(__file__))


for video in os.listdir(directory):
    vidNum = 0
    if video.endswith(".mp4"):
        capture = cv2.VideoCapture(video)

        # background is static, so we are looking for changes over a static background
        object_detector = cv2.createBackgroundSubtractorMOG2()

        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([[1, 0, 1, 0],
                         [0, 1, 0, 1],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])
        kf.P *= 1e3
        kf.R = np.diag([0.1, 0.1])
        kf.Q = np.diag([1e-4, 1e-4, 1e-4, 1e-4])

        trajectory = []

        while True:
            ret, frame = capture.read()

            # object detection
            mask = object_detector.apply(frame)

            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                # calculating area and removing the small/static elements
                area = cv2.contourArea(cnt)
                if area > 75:
                    x, y, w, h = cv2.boundingRect(cnt)

                    kf.predict()

                    measurement = np.array(([x + w / 2, y + h / 2]))
                    kf.update(measurement)
                    trajectory.append((int(kf.x[0]), int(kf.x[1])))
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if trajectory:
                cv2.polylines(frame, [np.array(trajectory)], False, (0, 255, 0), 2)

            # Video 1 display
            cv2.imshow("Frame", frame)
            # cv2.imshow('predictions', predictions)
            # cv2.imshow("Mask", mask)
            key = cv2.waitKey(27)
            if key == 27:
                break

        capture.release()
        cv2.destroyAllWindows()
