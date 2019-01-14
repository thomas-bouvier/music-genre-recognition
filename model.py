from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Input
from keras.layers import GRU
from keras.layers import concatenate
from keras.models import Model
from keras.applications.vgg16 import VGG16

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


def create_rcnn(input_shape, nb_genre):
    """
    Parallel recurrent neural network, used as test
    :param input_shape: Shape of the input of the network
    :param nb_genre: Number of genre
    :return: the model
    """
    input = Input(shape=input_shape)

    # Convolution layers
    convolution = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(input)
    convolution = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(convolution)
    convolution = MaxPooling1D(pool_size=2)(convolution)
    convolution = Dropout(CONV_DROPOUT_RATE)(convolution)
    convolution = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(convolution)
    convolution = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(convolution)
    convolution = MaxPooling1D(pool_size=2)(convolution)
    convolution = Dropout(CONV_DROPOUT_RATE)(convolution)
    convolution = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(convolution)
    convolution = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(convolution)
    convolution = MaxPooling1D(pool_size=2)(convolution)
    convolution = Dropout(CONV_DROPOUT_RATE)(convolution)
    convolution = Flatten()(convolution)

    # Recurrent layers
    recurrent = MaxPooling1D(pool_size=2)(input)
    recurrent = GRU(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(recurrent)
    recurrent = GRU(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(recurrent)
    recurrent = GRU(16, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(recurrent)

    # Create parallelism
    merged = concatenate([convolution, recurrent], axis=1)

    # Fully-connected layers
    out = Dense(512, activation='relu')(merged)
    out = Dense(512, activation='relu')(out)
    out = Dense(nb_genre, activation='softmax')(out)

    model = Model(input, out)

    return model


def create_model_vgg(input_shape, nb_genre):
    """
    Create a VGG16-like model using 2D convolutions.
    Used for testing.
    :param input_shape: Shape of the input
    :param nb_genre: Number of genre
    :return: the model
    """
    # Create VGG16
    vgg16 = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    output_shape = vgg16.output_shape
    output = vgg16.output

    # Add fully-connected layers
    top = Sequential()
    top.add(Flatten(input_shape=output_shape[1:]))
    top.add(Dense(256, activation='relu'))
    top.add(Dropout(DROPOUT_RATE))
    top.add(Dense(nb_genre, activation='softmax'))

    # Create model
    model = Model(inputs=vgg16.input, outputs=top(output))

    # Skip already trained layers
    for layer in model.layers[:5]:
        layer.trainable = False

    return model


def create_model_old(input_shape, nb_genre):
    """
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

