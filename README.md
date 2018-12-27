# music-genre-recognition
## Introduction
Music gere recognition algorithm using Convolutional Neural Network with the VGG16 model.

It works by splitting a sound in multiple overlapping windows, then the mel spectrogram of each
part of the song is created and reshaped to 3 channels to act as an image.

2D convolutions are applied by the model, and the genre of the music is determined.
Accuracy is around 80% at the moment.
## Dataset
Download the GTZAN dataset [here](http://opihi.cs.uvic.ca/sound/genres.tar.gz) and extract it in a **musics** folder.

You can also get already formated data for the CNN and the trained model [here](https://mega.nz/#!3rhxWaDS!6sgy0BzETQSt5ERZBjjYwWEdBiCk7qm4PKnzo8qCldU), and simply extract the files. 

## Requirements
The following modules can be installed via pip or using an IDE like PyCharm:
 - Python v3.6 - https://www.python.org/downloads/
 - Numpy
 - Keras
 - Tensorflow CPU or GPU (recommended for training) - https://www.tensorflow.org/install/
    - If you want to use the GPU version, you will need CUDA v9 and other requirements - https://www.tensorflow.org/install/gpu
 - Librosa
 - Sklearn
 - An audio backend like ffmpeg - https://ffmpeg.org/

## How to run
### Train
To train the model, follow these steps:
1. If you want to use already formatted data, set `LOAD_DATA_FROM_FILE = True` in `test.py`.
Otherwise, set it to `False` but it will take longer time and use a lot of RAM.
2. Set `TRAIN_MODE = True`.
3. Run `test.py` and wait.

The files `data_x.npy` and `data_y.npy` store the already formatted data.

Note that it can take a long time, especially with Tensorflow CPU, and a lot of RAM will be used - aroung 5Gb.
### Test
**You will need a trained model to test.**

To test the model against your own audio file, follow these steps:
1. Put a 30s audio file in the folder
2. Open `test.py` and change the following things:
    ```
    TRAIN_MODE = False
    TEST_FILE = './your-file.mp3'
    ```
3. Run `test.py`, the genre of the song will be outputted at the end.