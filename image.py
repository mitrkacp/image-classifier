import keras
import cv2
import numpy as np
import os
from image_classification_tools.classifier import Classifier

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def main():
    print(DIR_PATH)
    # initialize Classifier object
    classifier = Classifier(DIR_PATH)

    # read image from file
    img = cv2.imread("img.png")
    img = cv2.resize(img, (224, 224))

    label = classifier.class_detect(img)

    cv2.putText(
        img=img,
        text=label,
        org=(25, 25),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1.0,
        color=(0, 0, 255),
        thickness=2
    )
    cv2.imshow("Image", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()


