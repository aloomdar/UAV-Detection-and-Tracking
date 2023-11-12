import cv2
import os

directory = os.path.dirname(os.path.abspath(__file__))

for video in os.listdir(directory):
    vidNum = 0
    if video.endswith(".mp4"):
        vidNum += 1
        picNum = 1
        capture = cv2.VideoCapture(video)

        # background is static, so we are looking for changes over a static background
        object_detector = cv2.createBackgroundSubtractorMOG2()

        while True:
            ret, frame = capture.read()

            # region of interest
            # roi = frame[0: 700, 0: 1280]
            # applying this to mask breaks the for loop and doesn't go through the whole video

            # object detection
            mask = object_detector.apply(frame)

            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                # calculating area and removing the small/static elements
                area1 = cv2.contourArea(cnt)
                if area1 > 75:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.imwrite(directory + r'\detections\Video ' f'{vidNum}' ' picture ' f'{picNum}' '.jpeg', frame)
                    picNum += 1
                    # cv2.drawContours(frame1, [cnt], -1, (0, 255, 0), 2)

            # Video 1 display
            # cv2.imshow("Frame", frame)
            # cv2.imshow("Mask", mask)
            key = cv2.waitKey(30)
            if key == 27:
                break

        capture.release()
        cv2.destroyAllWindows()
