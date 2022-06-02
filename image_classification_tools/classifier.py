import tensorflow.keras
import cv2
import numpy as np
import os

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class Classifier:
    def __init__(self, path):
        self.basepath = path
        self.labels_path = f"{path}/dataset/labels.txt"
        self.model_path = f"{path}/model/keras_model2.h5"
        self.classes = self.read_labels(self.labels_path)
        self.model = tensorflow.keras.models.load_model(self.model_path)

    def read_labels(self, path):
        # open input file label.txt
        labelsfile = open(path, 'r')

        # initialize classes and read in all lines
        classes = []
        line = labelsfile.readline()
        while line:
            # read class name and append to classes
            classes.append(line.split(' ', 1)[1].rstrip())
            line = labelsfile.readline()
        # close labels file
        labelsfile.close()
        return classes

    def class_detect(self, img, ):
        # resize image
        size = (224, 224)
        img = cv2.resize(img, size)
        # convert image to array
        image_array = np.asarray(img)

        # normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        normalized_image_array = np.resize(normalized_image_array, (224, 224, 3))

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # load the image into the array
        data[0] = normalized_image_array
        # run the inference
        prediction = self.model.predict(data)
        class_number = np.argmax(prediction)

        confidence = []

        for i in range(0, len(self.classes)):
            # scale prediction confidence to % and apppend to list
            confidence.append(int(prediction[0][i] * 100))

        label = self.classes[class_number] + " " + str(confidence[class_number]) + "%"

        # cv2.putText(img, label, (25, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        return label
        # cv2.putText(
        #     img=img,
        #     text=label,
        #     org=(25, 25),
        #     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #     fontScale=1.0,
        #     color=(0, 255, 0)
        # )
        # cv2.imshow("Image", img)
        # cv2.waitKey(0)