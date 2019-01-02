# music-genre-recognition
## Introduction
Music genre recognition algorithm using Convolutional Neural Network with the VGG16 model.

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
 - H5py
 - Matplotlib

## How to run
### Train
To train the model you can run the following command:
```
python main.py -m train
```
This will load the audio files, format it and train the model.

Note that it can take a long time, especially with Tensorflow CPU and it will require some RAM.

You can also add the following parameters to the command:
- `--no-save-data` Once the data is loaded and formatted, it is saved as a .npy file so that you don't need to do that part again. With this flag, the data won't be saved.
- `--no-save-model` The trained model is saved in a .h5 file. With this flag, the model is not saved.
- `--load-data` If you already have .npy files with the formatted data, this flag will load them instead of loading data from audio files. Data won't be saved again afterwards.
- `--debug` Enable debug mode (shows more information).
### Test
**You will need a trained model to test.**

Once you have your model and your audio file, put them in the root folder.

To test the model against your own audio file, run the following command:
```
python main.py -m test -song your_song.mp3
```
This will load the file, process it and run it through the model.

You can add the following parameter:
- `--debug` Enable debug mode (shows more information).
## Results
Accuracy of the model is 82%, however it works best with old songs (before 2000) since the dataset used
to train it was created around this time. Since music has evolved, it will make more mistakes on modern songs.