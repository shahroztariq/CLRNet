import csv
import glob
import multiprocessing
import platform
import shutil
import sys
from time import strftime, gmtime

from keras.callbacks import ModelCheckpoint, CSVLogger
from keras_preprocessing.image import ImageDataGenerator
from pandas import DataFrame, read_csv
import pandas as pd
import numpy as np
import keras
from PIL import Image
from keras import Sequential
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.python.client import device_lib
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Activation, Average, BatchNormalization, \
    Flatten, Dense, regularizers, Convolution2D, UpSampling2D, LeakyReLU
from keras import Input
from keras.models import Model
from keras.layers import average
import os
import matplotlib.pyplot as plt
import imgaug as ia
from imgaug import augmenters as iaa
import cv2
from keras.applications.nasnet import preprocess_input
# import matplotlib.pyplot as plt
from auroc import read_file, compute_auroc


class Imgaug_DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, images_paths, labels, batch_size=64, image_dimensions=(96, 96, 3), shuffle=False, augment=False):
        self.labels = labels  # array of labels
        self.images_paths = images_paths  # array of image paths
        self.dim = image_dimensions  # image dimensions
        self.batch_size = batch_size  # batch size
        self.shuffle = shuffle  # shuffle bool
        self.augment = augment  # augment data bool
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.images_paths) / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.images_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        # selects indices of data for next batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # select data and load images
        labels = np.array([self.labels[k] for k in indexes])

        images = [cv2.resize(cv2.imread(self.images_paths[k]), dsize=(self.dim[0], self.dim[0]), interpolation=cv2.INTER_LINEAR) for k in indexes]

        # preprocess and augment data
        if self.augment == True:
            images = self.augmentor(images)

        images = np.array([preprocess_input(img) for img in images])

        return images, labels

    def augmentor(self, images):
        'Apply data augmentation'
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        seq = iaa.Sequential(
            [
                # apply the following augmenters to most images
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                iaa.Flipud(0.2),  # vertically flip 20% of all images
                sometimes(iaa.Affine(
                    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                    # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    # translate by -20 to +20 percent (per axis)
                    rotate=(-10, 10),  # rotate by -45 to +45 degrees
                    shear=(-5, 5),  # shear by -16 to +16 degrees
                    order=[0, 1],
                    # use nearest neighbour or bilinear interpolation (fast)
                    cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                    mode=ia.ALL
                    # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                           [sometimes(iaa.Superpixels(p_replace=(0, 1.0),
                                                      n_segments=(20, 200))),
                            # convert images into their superpixel representation
                            iaa.OneOf([
                                iaa.GaussianBlur((0, 1.0)),
                                # blur images with a sigma between 0 and 3.0
                                iaa.AverageBlur(k=(3, 5)),
                                # blur image using local means with kernel sizes between 2 and 7
                                iaa.MedianBlur(k=(3, 5)),
                                # blur image using local medians with kernel sizes between 2 and 7
                            ]),
                            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.9, 1.1)),
                            # sharpen images
                            iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
                            # emboss images
                            # search either for all edges or for directed edges,
                            # blend the result with the original image using a blobby mask
                            iaa.SimplexNoiseAlpha(iaa.OneOf([
                                iaa.EdgeDetect(alpha=(0.5, 1.0)),
                                iaa.DirectedEdgeDetect(alpha=(0.5, 1.0),
                                                       direction=(0.0, 1.0)),
                            ])),
                            iaa.AdditiveGaussianNoise(loc=0,
                                                      scale=(0.0, 0.01 * 255),
                                                      per_channel=0.5),
                            # add gaussian noise to images
                            iaa.OneOf([
                                iaa.Dropout((0.01, 0.05), per_channel=0.5),
                                # randomly remove up to 10% of the pixels
                                iaa.CoarseDropout((0.01, 0.03),
                                                  size_percent=(0.01, 0.02),
                                                  per_channel=0.2),
                            ]),
                            iaa.Invert(0.01, per_channel=True),
                            # invert color channels
                            iaa.Add((-2, 2), per_channel=0.5),
                            # change brightness of images (by -10 to 10 of original value)
                            iaa.AddToHueAndSaturation((-1, 1)),
                            # change hue and saturation
                            # either change the brightness of the whole image (sometimes
                            # per channel) or change the brightness of subareas
                            iaa.OneOf([
                                iaa.Multiply((0.9, 1.1), per_channel=0.5),
                                iaa.FrequencyNoiseAlpha(
                                    exponent=(-1, 0),
                                    first=iaa.Multiply((0.9, 1.1),
                                                       per_channel=True),
                                    second=iaa.ContrastNormalization(
                                        (0.9, 1.1))
                                )
                            ]),
                            sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5),
                                                                sigma=0.25)),
                            # move pixels locally around (with random strengths)
                            sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                            # sometimes move parts of the image around
                            sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                            ],
                           random_order=True
                           )
            ],
            random_order=True
        )
        return seq.augment_images(images)

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, labels, batch_size=32, dim=(32, 32), n_channels=3,
                 n_classes=2, shuffle=True, directory='datasets', gpu=1, epochs=500, image_type="jpg", cpu=1,
                 period_checkpoint=10, file_name='youdidntset', device_name='nil', checkpoint_dir="nil",
                 target_size=32, resize=False, optimizer='adam', loss='binary_crossentropy', class_mode='categorical'):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.directory = directory
        self.gpu = gpu,
        self.device_name = device_name
        self.cpu = cpu
        self.checkpoint_dir = checkpoint_dir
        self.file_name = file_name
        self.epochs = epochs
        self.image_type = image_type
        self.period_checkpoint = period_checkpoint
        self.target_size = target_size
        self.resize = resize
        self.optimizer = optimizer
        self.loss = loss
        self.class_mode = class_mode
        # print("params", batch_size, dim, n_channels, n_classes, shuffle, directory)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # X : (n_samples, *dim, n_channels)
        # Initialization
        if self.resize is True:
            X = np.empty((self.batch_size, *(self.target_size, self.target_size), self.n_channels))
        else:
            X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store class
            y[i] = self.labels[ID]
            # Store sample
            if self.resize is True:
                X[i,] = np.array(Image.open(ID).resize((self.target_size, self.target_size), Image.ANTIALIAS)) / 255
            else:
                X[i,] = np.array(Image.open(ID)) / 255

                # print(X[i].shape)
        # print(y,keras.utils.to_categorical(y, num_classes=self.n_classes))
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def make_partition(path, img_rows=32, img_cols=32, img_type="jpg", checkpoint_dir="nil"):
    # path: dataset folder
    # img_size: the size of images to be used
    # img_type: jpg, png etc
    fake_path = "{}/{}x{}/test/fake".format(path, img_rows, img_cols)
    fake_imgs = glob.glob(fake_path + '/*.' + img_type)
    fake_path = "{}/{}x{}/train/fake".format(path, img_rows, img_cols)
    fake_imgs += glob.glob(fake_path + '/*.' + img_type)
    fake_imgs = fake_imgs[:30000]

    real_path = "{}/{}x{}/test/real".format(path, img_rows, img_cols)
    real_imgs = glob.glob(real_path + '/*.' + img_type)
    real_path = "{}/{}x{}/train/real".format(path, img_rows, img_cols)
    real_imgs += glob.glob(real_path + '/*.' + img_type)

    partition = dict()
    label = dict()
    partition['X_train'] = []
    partition['X_test'] = []
    partition['y_train'] = []
    partition['y_test'] = []
    partition['X_validation'] = []
    partition['y_validation'] = []
    X, y = [], []
    for id in real_imgs:
        label[id] = 1
        X.append(id)
        y.append(1)
    for id in fake_imgs:
        label[id] = 0
        X.append(id)
        y.append(0)
    # partition['train'], partition['validation'], y_train, y_test = train_test_split(X, y, test_size=0.20,
    #                                                                                 random_state=42)
    partition['X_train'], partition['X_validation'], partition['y_train'], partition['y_validation'] = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42)

    #    # partition['X_train'], partition['X_test'], partition['y_train'], partition['y_test'] = train_test_split(
    #    #     partition['X_train'], partition['y_train'], test_size=0.10,
    #    #     random_state=42)
    file = open(os.path.join(checkpoint_dir, 'datasetsplit'), "w")
    file.write("Real Images: " + str(len(real_imgs)))
    file.write("\nFake Images: " + str(len(fake_imgs)))
    file.write("\nTrain Size: " + str(len(partition['X_train'])))
    file.write("\nValidation Size: " + str(len(partition['X_validation'])))
    file.write("\nTest Size: " + str(len(partition['X_test'])))
    file.close()
    print("Real Images:", len(real_imgs))
    print("Fake Images:", len(fake_imgs))
    print("Train Size:", len(partition['X_train']))
    print("Validation Size:", len(partition['X_validation']))
    print("Test Size:", len(partition['X_test']))
    # print(len(partition),len(partition['X_train'])+len(partition['X_validation']),partition['X_train'][0],partition['X_validation'][0],partition)
    return partition, label


def getAllGenerators(params):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255, validation_split=0.1)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    training_generator = train_datagen.flow_from_directory(
        params['train_directory'],  # this is the target directory
        target_size=params['dim'],  # all images will be resized to dim
        batch_size=params['batch_size'],
        class_mode=params['class_mode'],
        subset='training')
    validation_generator = train_datagen.flow_from_directory(
        params['train_directory'],
        target_size=params['dim'],
        batch_size=params['batch_size'],
        class_mode=params['class_mode'],
        subset='validation')
    test_generator = test_datagen.flow_from_directory(
        params['test_directory'],
        target_size=params['dim'],
        shuffle=False,
        batch_size=1,
        class_mode=params['class_mode'])
    return training_generator, validation_generator, test_generator


def createParams(img_size):
    current_file_name = os.path.basename(__file__)[:-3]
    params = {'dim': (img_size, img_size),
              'n_channels': 3,
              'batch_size': 5,
              'gpu': len(get_available_gpus()),
              'device_name': platform.node(),
              'cpu': multiprocessing.cpu_count(),
              'epochs': 50,
              'checkpoint_dir': os.path.join('checkpoints', strftime("%Y-%m-%d-%H-%M-%S", gmtime())),
              'file_name': current_file_name,
              'n_classes': 2,
              'shuffle': True,
              'train_directory': "datasets/{}x{}/train".format(img_size, img_size),
              'test_directory': "datasets/{}x{}/test".format(img_size, img_size),
              'class_mode': 'categorical',
              'image_type': 'jpg',
              'period_checkpoint': 1,
              'optimizer': 'adam',
              'loss': 'binary_crossentropy'}
    if not os.path.exists(params['checkpoint_dir']):
        os.makedirs(strftime(params['checkpoint_dir']))
    print(params)
    return params


def modelCompileFit(params, model, training_generator, validation_generator):
    checkpoints = ModelCheckpoint(
        os.path.join(params['checkpoint_dir'], params['file_name'] + '_{epoch:02d}' + '.hd5f'),
        save_weights_only=True,
        period=params['period_checkpoint'])
    csv_logger = CSVLogger(os.path.join(params['checkpoint_dir'], 'log.csv'), append=True, separator=',')
    file = open(os.path.join(params['checkpoint_dir'], 'parameters.txt'), "w")
    file.write(str(params))
    file.write("\nTraining:")
    file.write(str(len(training_generator.filenames)))
    file.write("\nValidation:")
    file.write(str(len(validation_generator.filenames)))
    file.write("\n")
    model.summary(print_fn=lambda x: file.write(x + '\n'))
    file.close()
    model.compile(loss=params['loss'],
                  optimizer=params['optimizer'],
                  metrics=['accuracy'])
    # Train model on dataset
    trained_model = model.fit_generator(generator=training_generator,
                                        steps_per_epoch=len(training_generator.filenames) // params['batch_size'],
                                        validation_data=validation_generator,
                                        validation_steps=len(validation_generator.filenames) // params['batch_size'],
                                        epochs=params['epochs'],
                                        callbacks=[checkpoints, csv_logger],
                                        verbose=1, use_multiprocessing=True, workers=params['cpu'])
    file = open(os.path.join(params['checkpoint_dir'], 'history.txt'), "w")
    file.write(str(trained_model.history))
    file.close()
    print(trained_model.history)
    return trained_model


def testModel(params, model, test_generator):
    test_generator.reset()
    y_pred = model.predict_generator(generator=test_generator,
                                     steps=len(test_generator.filenames) // 1,  # has to be divisible, 1 is best option
                                     verbose=0)
    np.set_printoptions(formatter={'float_kind': '{:.04f}'.format})
    real_y_pred = y_pred
    y_pred = np.argmax(y_pred, axis=1)
    y_true = test_generator.classes
    report = classification_report(y_true, y_pred)
    overall_accuracy = accuracy_score(y_true, y_pred)
    sk_auc = metrics.roc_auc_score(y_true, y_pred)
    print(report)
    print('Overall Accuracy:', overall_accuracy)
    print("SK_AUC:", sk_auc)
    names = np.array([os.path.basename(os.path.splitext(name)[0]) for name in test_generator.filenames])
    predict_file = os.path.join(params['checkpoint_dir'], 'predicition_label.txt')
    predictions = np.column_stack((names, real_y_pred[:, 0]))  # 0 for true; 1 for opposite result
    with open(predict_file, "w") as myfile:
        for item in predictions:
            myfile.write("{},{:1.4f}\n".format(item[0], float(item[1])))
    [predict, target] = read_file(predict_file, params['label_file'])
    [auc, roc] = compute_auroc(predict, target)
    print('Computed AUROC score: ', auc)
    n = len(predict)
    # This is for plotting
    plt.plot(*zip(*roc), label='ROC curve')
    plt.plot([0, 1], label='Random guess', linestyle='--', color='red')
    plt.legend(loc=4)
    plt.ylabel('TPR (True Positive Rate)')
    plt.xlabel('FPR (False Positive Rate)')
    plt.title('ROC Curve (AUROC : %7.3f)' % (auc))
    plt.axis([0, 1, 0, 1])
    plt.grid()
    plotname = os.path.join(params['checkpoint_dir'], 'plot.png')
    test_results = os.path.join(params['checkpoint_dir'], 'test_result.csv')
    classification_reports = os.path.join(params['checkpoint_dir'], 'classification_reports.txt')
    plt.savefig(plotname)
    plt.clf()
    with open(test_results, "a") as myfile:
        myfile.write(str("{},{},{},{}\n".format(os.path.basename(params['checkpoint_dir']), overall_accuracy, sk_auc, auc)))
    with open(classification_reports, "a") as myfile:
        myfile.write(
            str("{}\n{}{}\n{}\n{}\n\n".format(os.path.basename(params['checkpoint_dir']), report, overall_accuracy, sk_auc,
                                              auc)))
    #keras.backend.clear_session()

def ensembleModels(models, model_input):
    if len(models) == 1:
        return models[0]
    # collect outputs of models in a list
    yModels = [model(model_input) for model in models]
    # averaging outputs
    yAvg = average(yModels)
    # build model from same input and avg output
    modelEns = Model(inputs=model_input, outputs=yAvg, name='ensemble')

    return modelEns


def load_3CB1D(weights, name,shape):
    model = _3CB1D(shape)
    model.load_weights(weights)
    model.name = name
    return model


def _3CB1D(input_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))
    return model


def load_shallow_cnn(weights, name, model_input):
    model = shallow_cnn(model_input)
    model.load_weights(weights)
    model.name = name
    return model

# ShallowNetV2
def shallow_cnn(model_input):
    # Block1
    x = Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0001))(
        model_input)
    x = Dropout(0.25)(x)
    x = Conv2D(96, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(96, (3, 3), activation='relu', padding='same', strides=2, kernel_regularizer=regularizers.l2(0.0001))(x)
    x = Dropout(0.25)(x)
    # Block2
    x = Conv2D(192, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same', strides=2, kernel_regularizer=regularizers.l2(0.0001))(x)
    x = Dropout(0.25)(x)
    # Block3
    x = Conv2D(192, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(192, (1, 1), activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = Dropout(0.25)(x)
    # Block4
    x = Flatten()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Dense(2, activation='sigmoid')(x)

    # x = Conv2D(2, (1, 1))(x)
    # x = GlobalAveragePooling2D()(x)
    # x = Activation(activation='softmax')(x)
    model = Model(model_input, x, name='shallow_cnn')
    return model


def load_cpd64(weights, name, model_input):
    model = cpd64(model_input)
    model.load_weights(weights)
    model.name = name
    return model


def cpd64(input_shape):
    model = Sequential()
    # Block 1
    model.add(Conv2D(96, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C1',
                     input_shape=input_shape))
    model.add(BatchNormalization(name='B1'))
    model.add(Activation('relu', name='A1'))
    model.add(Dropout(0.25, name='O1'))
    model.add(Conv2D(96, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C2'))
    model.add(BatchNormalization(name='B2'))
    model.add(Activation('relu', name='A2'))
    model.add(Dropout(0.25, name='O2'))
    model.add(Conv2D(96, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C3'))
    model.add(BatchNormalization(name='B3'))
    model.add(Activation('relu', name='A3'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, name='P1'))
    model.add(Dropout(0.25, name='O3'))
    # Block 2
    model.add(Conv2D(192, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C4'))
    model.add(BatchNormalization(name='B4'))
    model.add(Activation('relu', name='A4'))
    model.add(Dropout(0.25, name='O4'))
    model.add(Conv2D(192, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C5'))
    model.add(BatchNormalization(name='B5'))
    model.add(Activation('relu', name='A5'))
    model.add(Dropout(0.25, name='O5'))
    model.add(Conv2D(192, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C6'))
    model.add(BatchNormalization(name='B6'))
    model.add(Activation('relu', name='A6'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, name='P2'))
    model.add(Dropout(0.25, name='O6'))
    # Block 3
    model.add(Conv2D(192, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C7'))
    model.add(BatchNormalization(name='B7'))
    model.add(Activation('relu', name='A7'))
    model.add(Dropout(0.25, name='O7'))
    model.add(Conv2D(192, (1, 1), padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C8'))
    model.add(BatchNormalization(name='B8'))
    model.add(Activation('relu', name='A8'))
    model.add(Dropout(0.25, name='O8'))
    # Block 4
    model.add(Flatten())
    model.add(Dense(1024, kernel_regularizer=regularizers.l2(0.0001), name='D1'))
    model.add(BatchNormalization(name='B9'))
    model.add(Activation('relu', name='A9'))
    model.add(Dropout(0.25, name='O9'))
    model.add(Dense(2, activation='sigmoid'))
    model.name = 'cpd64'
    return model


def load_cpd128(weights, name, model_input):
    model = cpd128(model_input)
    model.load_weights(weights)
    model.name = name
    return model

# ShallowNet V1
def cpd128(input_shape):
    model = Sequential()
    # Block 1
    model.add(Conv2D(96, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C1',
                     input_shape=input_shape))
    model.add(BatchNormalization(name='B1'))
    model.add(Activation('relu', name='A1'))
    model.add(Dropout(0.25, name='O1'))
    model.add(Conv2D(96, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C2'))
    model.add(BatchNormalization(name='B2'))
    model.add(Activation('relu', name='A2'))
    model.add(Dropout(0.25, name='O2'))
    model.add(Conv2D(96, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C3'))
    model.add(BatchNormalization(name='B3'))
    model.add(Activation('relu', name='A3'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, name='P1'))
    model.add(Dropout(0.25, name='O3'))
    # Block 2
    model.add(Conv2D(192, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C4'))
    model.add(BatchNormalization(name='B4'))
    model.add(Activation('relu', name='A4'))
    model.add(Dropout(0.25, name='O4'))
    model.add(Conv2D(192, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C5'))
    model.add(BatchNormalization(name='B5'))
    model.add(Activation('relu', name='A5'))
    model.add(Dropout(0.25, name='O5'))
    model.add(Conv2D(192, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C6'))
    model.add(BatchNormalization(name='B6'))
    model.add(Activation('relu', name='A6'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, name='P2'))
    model.add(Dropout(0.25, name='O6'))
    # Block 3
    model.add(Conv2D(257, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C7'))
    model.add(BatchNormalization(name='B7'))
    model.add(Activation('relu', name='A7'))
    model.add(Dropout(0.25, name='O7'))
    model.add(Conv2D(257, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C8'))
    model.add(BatchNormalization(name='B8'))
    model.add(Activation('relu', name='A8'))
    model.add(Dropout(0.25, name='O8'))
    model.add(Conv2D(257, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C9'))
    model.add(BatchNormalization(name='B9'))
    model.add(Activation('relu', name='A9'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, name='P3'))
    model.add(Dropout(0.25, name='O9'))
    # Block 4
    model.add(Conv2D(311, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C10'))
    model.add(BatchNormalization(name='B10'))
    model.add(Activation('relu', name='A10'))
    model.add(Dropout(0.25, name='O10'))
    model.add(Conv2D(311, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C11'))
    model.add(BatchNormalization(name='B11'))
    model.add(Activation('relu', name='A11'))
    model.add(Dropout(0.25, name='O11'))
    model.add(Conv2D(311, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C12'))
    model.add(BatchNormalization(name='B12'))
    model.add(Activation('relu', name='A12'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, name='P4'))
    model.add(Dropout(0.25, name='O12'))
    # Block 5
    model.add(Conv2D(396, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C13'))
    model.add(BatchNormalization(name='B13'))
    model.add(Activation('relu', name='A13'))
    model.add(Dropout(0.25, name='O13'))
    model.add(Conv2D(396, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C14'))
    model.add(BatchNormalization(name='B14'))
    model.add(Activation('relu', name='A14'))
    model.add(Dropout(0.25, name='O14'))
    model.add(Conv2D(396, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C15'))
    model.add(BatchNormalization(name='B15'))
    model.add(Activation('relu', name='A15'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, name='P5'))
    model.add(Dropout(0.25, name='O15'))
    # Block 6
    model.add(Conv2D(437, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C16'))
    model.add(BatchNormalization(name='B16'))
    model.add(Activation('relu', name='A16'))
    model.add(Dropout(0.25, name='O16'))
    model.add(Conv2D(437, (1, 1), padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C17'))
    model.add(BatchNormalization(name='B17'))
    model.add(Activation('relu', name='A17'))
    model.add(Dropout(0.25, name='O17'))
    # Block 7
    model.add(Flatten(name='F1'))
    model.add(Dense(3933, kernel_regularizer=regularizers.l2(0.0001), name='D1'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25, name='O18'))
    model.add(Dense(2, activation='sigmoid'))
    model.name = 'cpd128'
    return model


def load_nincnn(weights, name, model_input):
    model = nincnn(model_input)
    model.load_weights(weights)
    model.name = name
    return model

#ShallowNet V3
def nincnn(model_input):
    # block 1
    x = Conv2D(32, (5, 5), activation='relu', padding='valid', kernel_regularizer=regularizers.l2(0.0001))(model_input)
    x = Dropout(0.25)(x)
    x = Conv2D(32, (1, 1), activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(32, (1, 1), activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    # block 2
    x = Conv2D(64, (3, 3), activation='relu', padding='valid', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(64, (1, 1), activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(64, (1, 1), activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    # block 3
    x = Conv2D(128, (3, 3), activation='relu', padding='valid', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(32, (1, 1), activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = Dropout(0.25)(x)
    # block 4
    x = Flatten()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Dense(2, activation='sigmoid')(x)
    model = Model(model_input, x, name='nincnn')
    return model


def load_cpd64_old(weights, name, model_input):
    model = cpd64_old(model_input)
    model.load_weights(weights)
    model.name = name
    return model


def cpd64_old(input_shape):
    model = Sequential()
    model.add(
        Conv2D(96, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C1',
               input_shape=input_shape))
    model.add(Dropout(0.25, name='O1'))
    model.add(
        Conv2D(96, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C2'))
    model.add(Dropout(0.25, name='O2'))
    model.add(
        Conv2D(96, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C3'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, name='P1'))
    model.add(Dropout(0.25, name='O3'))
    model.add(
        Conv2D(192, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C4'))
    model.add(Dropout(0.25, name='O4'))
    model.add(
        Conv2D(192, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C5'))
    model.add(Dropout(0.25, name='O5'))
    model.add(
        Conv2D(192, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C6'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, name='P2'))
    model.add(Dropout(0.25, name='O6'))
    model.add(
        Conv2D(192, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C7'))
    model.add(Dropout(0.25, name='O7'))
    model.add(
        Conv2D(192, (1, 1), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0001), name='C8'))
    model.add(Dropout(0.25, name='O8'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0001), name='D1'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25, name='O9'))
    model.add(Dense(2, activation='sigmoid'))
    model.name = 'cpd64'
    return model


def load_cpd_old(weights, name, model_input):
    model = cpd_old(model_input)
    model.load_weights(weights)
    model.name = name
    return model


def cpd_old(model_input):
    x = Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0001))(
        model_input)
    x = Dropout(0.25)(x)
    x = Conv2D(96, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(96, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    x = Dropout(0.25)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    x = Dropout(0.25)(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(192, (1, 1), activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Dense(2, activation='sigmoid')(x)
    model = Model(model_input, x, name='cpd')
    return model


def sortfile(filename):
    reader = csv.reader(open(filename), delimiter=",")
    sortedlist = sorted(reader, key=lambda row: row[0], reverse=False)
    with open(filename, "w") as myfile:
        for item in sortedlist:
            myfile.write("{},{:1.4f}\n".format(item[0], float(item[1])))


def sortfile_org(filename):
    with open(filename, "r") as myfile:
        data = [r.strip().split(',') for r in myfile.readlines()]
    files = dict()
    for item in data:
        files[item[0]] = item[1]
    print(sorted(files))
    print(len(sorted(files)))
    with open(filename + '1', "w") as myfile:
        for item in data:
            myfile.write("{},{}\n".format(item[0], item[1]))


def create_test_directory():
    if os.path.exists('test_data'):
        shutil.rmtree('test_data')
    os.makedirs('test_data')
    if os.path.exists(os.path.join('test_data', '64', 'imgs')):
        shutil.rmtree(os.path.join('test_data', '64', 'imgs'))
    os.makedirs(os.path.join('test_data', '64', 'imgs'))
    if os.path.exists(os.path.join('test_data', '128', 'imgs')):
        shutil.rmtree(os.path.join('test_data', '128', 'imgs'))
    os.makedirs(os.path.join('test_data', '128', 'imgs'))
    if os.path.exists(os.path.join('test_data', '256', 'imgs')):
        shutil.rmtree(os.path.join('test_data', '256', 'imgs'))
    os.makedirs(os.path.join('test_data', '256', 'imgs'))
    if os.path.exists(os.path.join('test_data', '1024', 'imgs')):
        shutil.rmtree(os.path.join('test_data', '1024', 'imgs'))
    os.makedirs(os.path.join('test_data', '1024', 'imgs'))


def divide_images_by_size(directory, img_type='jpg'):
    create_test_directory()
    images = glob.glob(directory + '/*.' + img_type)
    for image in images:
        with Image.open(image) as img:
            width, height = img.size
            if (width, height) == (64, 64):
                shutil.copy(image, os.path.join('test_data', '64', 'imgs'))
            if (width, height) == (128, 128):
                shutil.copy(image, os.path.join('test_data', '128', 'imgs'))
            if (width, height) == (256, 256):
                shutil.copy(image, os.path.join('test_data', '256', 'imgs'))
            if (width, height) == (1024, 1024):
                shutil.copy(image, os.path.join('test_data', '1024', 'imgs'))

def autoencoder(model_input):
    # Encoder
    x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(model_input)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
    encoded = MaxPooling2D((2, 2), border_mode='same')(x)

    # Decoder

    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)

    # In original tutorial, border_mode='same' was used.
    # then the shape of 'decoded' will be 32 x 32, instead of 28 x 28
    x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
    # x = Convolution2D(16, 3, 3, activation='relu', border_mode='valid')(x)

    x = UpSampling2D((2, 2))(x)
    decoded = Convolution2D(3, 5, 5, activation='sigmoid', border_mode='same')(x)
    autoencoder = Model(model_input, decoded)
    return autoencoder


def vqvae(model_input):
    # Encoder
    x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(model_input)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
    encoded = MaxPooling2D((2, 2), border_mode='same')(x)

    # Decoder

    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)

    # In original tutorial, border_mode='same' was used.
    # then the shape of 'decoded' will be 32 x 32, instead of 28 x 28
    x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
    # x = Convolution2D(16, 3, 3, activation='relu', border_mode='valid')(x)

    x = UpSampling2D((2, 2))(x)
    decoded = Convolution2D(3, 5, 5, activation='sigmoid', border_mode='same')(x)
    autoencoder = Model(model_input, decoded)
    return autoencoder


def meso4(model_input):
    # x = Input(shape=(IMGWIDTH, IMGWIDTH, 3))
    x = model_input
    x1 = Conv2D(8, (3, 3), padding='same', activation='relu')(model_input)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)

    x2 = Conv2D(8, (5, 5), padding='same', activation='relu')(x1)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)

    x3 = Conv2D(16, (5, 5), padding='same', activation='relu')(x2)
    x3 = BatchNormalization()(x3)
    x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)

    x4 = Conv2D(16, (5, 5), padding='same', activation='relu')(x3)
    x4 = BatchNormalization()(x4)
    x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)

    y = Flatten()(x4)
    y = Dropout(0.5)(y)
    y = Dense(16)(y)
    y = LeakyReLU(alpha=0.1)(y)
    y = Dropout(0.5)(y)
    y = Dense(1, activation='sigmoid')(y)

    return Model(inputs=x, outputs=y)

