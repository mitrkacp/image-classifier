from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
# from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from keras.models import load_model
# Use the Image Data Generator to import the images from the dataset
from keras.preprocessing.image import ImageDataGenerator


class ModelTrainer:
    def __init__(self, path, img_size=[224, 224]):
        self.base_path = path
        self.image_size = img_size
        self.training_path = f"{path}\\dataset\\Train"
        self.validation_path = f"{path}\\dataset\\Test"
        self.base_model = InceptionV3(input_shape=self.image_size + [3], weights='imagenet', include_top=False)
        self.folders = glob(f"{path}\\dataset\\Test\\*")

    def train_model(self):

        # don't train existing weights
        for layer in self.base_model.layers:
            layer.trainable = False

        # useful for getting number of output classes
        folders = self.folders


        # our layers - you can add more if you want
        x = Flatten()(self.base_model.output)

        prediction = Dense(len(folders), activation='softmax')(x)


        # create a model object
        model = Model(inputs=self.base_model.input, outputs=prediction)

        # view the structure of the model
        model.summary()

        # tell the model what cost and optimization method to use
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        # Make sure you provide the same target size as initialied for the image size
        training_set = train_datagen.flow_from_directory(self.training_path,
                                                         target_size=(224, 224),
                                                         batch_size=8,
                                                         class_mode='categorical')

        test_set = test_datagen.flow_from_directory(self.validation_path,
                                                    target_size=(224, 224),
                                                    batch_size=8,
                                                    class_mode='categorical')

        # fit the model
        # Run the cell. It will take some time to execute
        print(len(training_set))
        print(len(test_set))
        r = model.fit_generator(
            training_set,
            validation_data=test_set,
            epochs=10,
            steps_per_epoch=len(training_set),
            validation_steps=len(test_set)
        )

        # plot the loss
        plt.plot(r.history['loss'], label='train loss')
        plt.plot(r.history['val_loss'], label='val loss')
        plt.legend()
        plt.show()
        # plt.savefig('LossVal_loss.png')

        # plot the accuracy
        plt.plot(r.history['accuracy'], label='train acc')
        plt.plot(r.history['val_accuracy'], label='val acc')
        plt.legend()
        plt.show()
        # plt.savefig('AccVal_acc.png')

        # save it as a h5 file

        model.save(self.base_path + '/model/output_model.h5')
