from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Conv2D
from keras.layers import MaxPooling2D

DROPOUT_RATE = 0.5
CONV_DROPOUT_RATE = 0.2


def create_model(input_shape, nb_genre):
    """
    Create the CNN used for music genre recognition.
    :param input_shape: Shape of the input of the network
    :param nb_genre: Number of output classes
    :return: The model
    """
    # Create model
    model = Sequential()

    # First block
    model.add(Conv1D(filters=16,
                     kernel_size=3,
                     input_shape=input_shape,
                     activation='relu',
                     padding='same'))
    model.add(Conv1D(filters=16,
                     kernel_size=3,
                     activation='relu',
                     padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(CONV_DROPOUT_RATE))

    # Second block
    model.add(Conv1D(filters=32,
                     kernel_size=3,
                     activation='relu',
                     padding='same'))
    model.add(Conv1D(filters=32,
                     kernel_size=3,
                     activation='relu',
                     padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(CONV_DROPOUT_RATE))

    # Third block
    model.add(Conv1D(filters=64,
                     kernel_size=3,
                     activation='relu',
                     padding='same'))
    model.add(Conv1D(filters=64,
                     kernel_size=3,
                     activation='relu',
                     padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(CONV_DROPOUT_RATE))

    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(DROPOUT_RATE))

    # Output
    model.add(Dense(nb_genre, activation='softmax'))

    return model


def create_model_old(input_shape, nb_genre):
    """
    DEPRECATED FUNCTION
    Create 2D convolution model.
    Used for testing.
    :param input_shape: input shape of the CNN
    :param nb_genre: the number of classes
    :return: the CNN
    """
    model = Sequential()
    model.add(
        Conv2D(filters=16, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
    model.add(Conv2D(filters=16, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(CONV_DROPOUT_RATE))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(CONV_DROPOUT_RATE))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(CONV_DROPOUT_RATE))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(nb_genre, activation='softmax'))

    return model




