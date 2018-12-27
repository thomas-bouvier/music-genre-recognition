from keras.models import Sequential
from keras.layers import Dense
from keras.models import Model
from keras.layers import Dropout
from keras.layers import Flatten
from keras.applications.vgg16 import VGG16


def create_model(input_shape):
    vgg16 = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    output_shape = vgg16.output_shape
    output = vgg16.output

    top = Sequential()
    top.add(Flatten(input_shape=output_shape[1:]))
    top.add(Dense(256, activation='relu'))
    top.add(Dropout(0.5))
    top.add(Dense(10, activation='softmax'))

    model = Model(inputs=vgg16.input, outputs=top(output))

    for layer in model.layers[:5]:
        layer.trainable = False

    return model
