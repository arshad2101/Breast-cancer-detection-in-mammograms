# Imports
import keras
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from keras.preprocessing.image import load_img
from IPython.display import display
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

test_batchsize = 5   
image_size = 224      
test_dir = 'dataset/test'

def test_datagenerator():
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(test_dir,
                              target_size=(image_size, image_size),
                              batch_size=test_batchsize,
                              class_mode='categorical',
                              shuffle=False)
 
    return test_dir, test_generator


def test():
    test_dir1, test_generator = test_datagenerator()

    print('loading trained model...')
    new_model = keras.models.load_model('trained_models/model.h5')
    print('loading complete')

    print('summary of loaded model')
    new_model.summary()

    ground_truth = test_generator.classes

    print('predicting on the test images...')

    prediction_start = time.clock()
    predictions = new_model.predict_generator(test_generator,
                                              steps=test_generator.samples / test_generator.batch_size,
                                              verbose=0)

    prediction_finish = time.clock()
    prediction_time = prediction_finish - prediction_start
    predicted_classes = np.argmax(predictions, axis=1)

    errors = np.where(predicted_classes != ground_truth)[0]
    print("No. of errors = {}/{}".format(len(errors), test_generator.samples))

    correct_predictions = np.where(predicted_classes == ground_truth)[0]
    print("No. of correct predictions = {}/{}".format(len(correct_predictions), test_generator.samples))

    print("Test Accuracy = {0:.2f}%".format(len(correct_predictions)*100/test_generator.samples))
    print("Predicted in {0:.3f} minutes!".format(prediction_time/60))

def Main():
    test()                 

if __name__ == '__main__':
    Main()