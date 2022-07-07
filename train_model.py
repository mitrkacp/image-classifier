
from image_classification_tools.model_trainer import ModelTrainer
import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def main():
    model_trainer = ModelTrainer(DIR_PATH, n_epochs=10, baseline_val_accuracy=0.9)
    model_trainer.train()




if __name__ == '__main__':
    main()

