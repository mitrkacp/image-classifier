import tensorflow.keras
import cv2
import numpy as np
import os
from image_classification_tools.classifier import Classifier

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def main():
    # initialize Classifier object
    classifier = Classifier(DIR_PATH)

    # initialize webcam video object
    cap = cv2.VideoCapture(0)  # For USB cam change it to 1

    # width & height of webcam video in pixels
    frameWidth = 224
    frameHeight = 224
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
    # enable auto gain
    cap.set(cv2.CAP_PROP_GAIN, 0)

    while True:
        success, img = cap.read()
        label = classifier.class_detect(img)

        cv2.putText(
            img=img,
            text=label,
            org=(25, 25),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color=(0, 255, 0)
        )
        cv2.imshow("Camera", img)
        cv2.waitKey(1)

        # Exit, when 'q' is pressed on the keyboard
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
