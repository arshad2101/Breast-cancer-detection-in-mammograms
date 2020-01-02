from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Activation, Dropout
from keras.models import Model
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from keras.models import Sequential
from  keras.applications import VGG16, ResNet50
from keras import backend as K
from keras import optimizers
import os
import errno
from matplotlib import pyplot as plt
import time

# test >> normal = 62 | abnormal - 34
# training >> normal =  145 | abnormal - 81

# NN Parameters
image_size = 224     
train_batchsize = 10  
epochs = 90

# Image Dataset Directory
train_dir = 'dataset/train'

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

# Train datagenerator
def train_datagenerator(train_batchsize):
    
    train_datagen = ImageDataGenerator(
              rescale=1 / 255.0,
              rotation_range=20,
              zoom_range=0.05,
              width_shift_range=0.05,
              height_shift_range=0.05,
              shear_range=0.05,
              horizontal_flip=True,
              fill_mode="nearest")

    train_generator = train_datagen.flow_from_directory(train_dir,
                                target_size=(image_size, image_size),
                                batch_size=train_batchsize,
                                class_mode='categorical')

    return train_generator

def vgg16_finetuned():

  vgg_conv = VGG16(weights='imagenet',
            include_top=False,
            input_shape=(image_size, image_size, 3))

  for layer in vgg_conv.layers[:-2]:
    layer.trainable = False

  model = Sequential()
  model.add(vgg_conv)
  model.add(Flatten())
  model.add(Dense(1024, activation='relu'))
  model.add(Dropout(0.4))
  model.add(Dense(1024, activation='relu'))
  model.add(Dropout(0.4))
  model.add(Dense(2, activation= 'sigmoid'))

  return model

def residual_network_tuned():

  resnet_conv = ResNet50(weights='imagenet',
            include_top=False,
            input_shape=(image_size, image_size, 3))

  for layer in resnet_conv.layers[:-1]:
    layer.trainable = False

  model = Sequential()
  model.add(resnet_conv)
  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(64, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(2, activation='sigmoid'))

  return model

def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 90:
        lr *= 0.5e-3
    elif epoch > 80:
        lr *= 1e-3
    elif epoch > 60:
        lr *= 1e-2
    elif epoch > 40:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def train(model):
    train_generator = train_datagenerator(train_batchsize)
    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['acc'])
    train_start = time.clock()
    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                  cooldown=0,
                                  patience=5,
                                  min_lr=0.5e-6)
    callbacks = [lr_reducer, lr_scheduler]

    print('Started training...')
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=train_generator.samples / train_generator.batch_size,
                                  epochs=epochs,
                                  verbose=1, 
                                  callbacks=callbacks)

    train_finish = time.clock()
    train_time = train_finish - train_start
    print('Training completed in {0:.3f} minutes!'.format(train_time / 60))

    print('Saving the trained model...')
    model.save('trained_models/model.h5')
    print("Saved trained model in 'traned_models/ folder'!")

    return model, history


def show_graphs(history):
    
    acc = history.history['acc']
    loss = history.history['loss']

    epochs1 = range(len(acc))

    plt.plot(epochs1, acc, 'b', label='Training acc')
    plt.title('Training accuracy')
    plt.legend()
    plt.savefig('Training accuracy')

    plt.figure()

    plt.plot(epochs1, loss, 'b', label='Training loss')
    plt.title('Training loss')
    plt.legend()
    plt.savefig('Training loss')
    plt.show()


def Main():
    model = residual_network_tuned()

    print("epochs, train_batchsize", epochs, train_batchsize)
    _, history = train(model)  

    show_graphs(history)

if __name__ == '__main__':
    Main()