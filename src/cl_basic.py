from tensorflow.python._pywrap_tensorflow_internal import Flatten
from tensorflow.python.keras import Input, regularizers, Model
from tensorflow.python.keras.layers import ConvLSTM2D, BatchNormalization, Dropout, Conv3D, AveragePooling3D, Reshape, Dense, Conv2D, MaxPooling2D, MaxPooling3D
from tensorflow.python.keras.regularizers import L1L2
from tensorflow.python.keras.utils.layer_utils import print_summary


def cl_basic(h, w, c, num_classes):
    L1L2(l1=0.0001, l2=0.0001)
    # model = Sequential()
    inputs = Input(shape=(None, h, w, c))
    x = ConvLSTM2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu',
                   return_sequences=True,
                   kernel_regularizer=regularizers.L1L2(l1=0.0001))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = ConvLSTM2D(filters=80, kernel_size=(3, 3), padding='same', activation='relu',
                   return_sequences=True, kernel_regularizer=regularizers.L1L2(l1=0.0001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = ConvLSTM2D(filters=120, kernel_size=(3, 3), padding='same', activation='relu',
                   return_sequences=True, kernel_regularizer=regularizers.L1L2(l1=0.0001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = ConvLSTM2D(filters=160, kernel_size=(3, 3), padding='same', activation='relu',
                   return_sequences=True, kernel_regularizer=regularizers.L1L2(l1=0.0001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = ConvLSTM2D(filters=200, kernel_size=(3, 3), padding='same', activation='relu',
                   return_sequences=True, kernel_regularizer=regularizers.L1L2(l1=0.0001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = ConvLSTM2D(filters=240, kernel_size=(3, 3), padding='same', activation='relu',
                   return_sequences=True, kernel_regularizer=regularizers.L1L2(l1=0.0001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu', padding='same',
               data_format='channels_last', kernel_regularizer=regularizers.L1L2(l1=0.0001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = AveragePooling3D((1, 128, 128))(x)
    # x=Flatten()(x)
    x = Reshape((-1, 256))(x)
    outputs = Dense(
        units=num_classes,
        activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    # model.build()

    return model

#ShallowNet V3
def nincnn(model_input):
    # block 1
    x = ConvLSTM2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu',
                   return_sequences=True,
                   kernel_regularizer=regularizers.L1L2(l1=0.0001))(model_input)
    x = Dropout(0.25)(x)
    x = ConvLSTM2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu',
                   return_sequences=True,
                   kernel_regularizer=regularizers.L1L2(l1=0.0001))(x)
    x = Dropout(0.25)(x)
    x = ConvLSTM2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu',
                   return_sequences=True,
                   kernel_regularizer=regularizers.L1L2(l1=0.0001))(x)
    x = MaxPooling3D((1, 2,2))(x)
    x = Dropout(0.25)(x)
    # block 2
    x = ConvLSTM2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu',
                   return_sequences=True,
                   kernel_regularizer=regularizers.L1L2(l1=0.0001))(x)
    x = Dropout(0.25)(x)
    x = ConvLSTM2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu',
                   return_sequences=True,
                   kernel_regularizer=regularizers.L1L2(l1=0.0001))(x)
    x = Dropout(0.25)(x)
    x = ConvLSTM2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu',
                   return_sequences=True,
                   kernel_regularizer=regularizers.L1L2(l1=0.0001))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    # block 3
    x = ConvLSTM2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu',
                   return_sequences=True,
                   kernel_regularizer=regularizers.L1L2(l1=0.0001))(x)
    x = Dropout(0.25)(x)
    x = ConvLSTM2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu',
                   return_sequences=True,
                   kernel_regularizer=regularizers.L1L2(l1=0.0001))(x)
    x = Dropout(0.25)(x)
    # block 4
    x = Flatten()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Dense(2, activation='sigmoid')(x)
    model = Model(model_input, x, name='nincnn')
    return model

#ShallowCLNET V3
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


