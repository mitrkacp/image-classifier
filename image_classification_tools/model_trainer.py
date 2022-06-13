from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
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
    def __init__(self, path, img_size=[224, 224], n_epochs=100):
        self.base_path = path
        self.image_size = img_size
        self.training_path = f"{path}\\dataset\\Train"
        self.validation_path = f"{path}\\dataset\\Validation"
        self.base_model = InceptionV3(input_shape=self.image_size + [3], weights='imagenet', include_top=False)
        self.classes = glob(f"{path}\\dataset\\Train\\*")
        self.epochs = n_epochs

    def train(self):
        # freeze existing weights to not train them
        for layer in self.base_model.layers:
            layer.trainable = False

        # add own output layers
        x = Flatten()(self.base_model.output)

        prediction = Dense(len(self.classes), activation='softmax')(x)
        print( f'x: {x}')
        print( f'prediction: {prediction}')

        # create a model object
        model = Model(inputs=self.base_model.input, outputs=prediction)

        # view the structure of the model
        model.summary()

        # compile model and set cost and optimization method
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        # generate more data
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        training_set = train_datagen.flow_from_directory(self.training_path,
                                                         target_size=(224, 224),
                                                         batch_size=8,
                                                         class_mode='categorical')

        validation_set = test_datagen.flow_from_directory(self.validation_path,
                                                    target_size=(224, 224),
                                                    batch_size=8,
                                                    class_mode='categorical')

        # fit the model
        result = model.fit_generator(
            training_set,
            validation_data=validation_set,
            epochs=self.epochs,
            steps_per_epoch=len(training_set),
            validation_steps=len(validation_set)
        )

        # plot the loss
        plt.plot(result.history['loss'], label='train loss')
        plt.plot(result.history['val_loss'], label='val loss')
        plt.legend()
        plt.show()
        # plt.savefig('LossVal_loss.png')

        # plot the accuracy
        plt.plot(result.history['accuracy'], label='train acc')
        plt.plot(result.history['val_accuracy'], label='val acc')
        plt.legend()
        plt.show()
        # plt.savefig('AccVal_acc.png')

        # save model as a h5 file
        model.save(self.base_path + '/model/output_model.h5')
