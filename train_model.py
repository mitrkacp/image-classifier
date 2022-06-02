
from image_classification_tools.model_trainer import ModelTrainer
import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def main():
    print(DIR_PATH)
    model_trainer = ModelTrainer(DIR_PATH)
    print(model_trainer.validation_path)
    print(model_trainer.training_path)
    print(model_trainer.folders)
    model_trainer.train_model()




if __name__ == '__main__':
    main()

