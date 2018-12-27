import librosa
import numpy as np
import os
from keras.utils import to_categorical
import random
from model import create_model
from keras.models import load_model
from sklearn.model_selection import train_test_split

GENRES = {'metal': 0, 'disco': 1, 'classical': 2, 'hiphop': 3, 'jazz': 4,
          'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'rock': 9}
NUM_GENRES = len(GENRES)
DATA_FOLDER = "./musics"
DEBUG = True
SAVE_DATA = False
LOAD_DATA_FROM_FILE = False
TRAIN_MODE = True
TEST_FILE = './sample.mp3'
DATA_X_FILE = './data_x.npy'
DATA_Y_FILE = './data_y.npy'
# Set to 660000 instead of 661500 because depending on the format, it can cause problems
SAMPLE_SIZE = 660000


# TODO: Try with 1D, different architecture


def main():
    """
    Main function of the program
    If training mode, load the data, train the model and save it
    If testing mode, load the music, load the model and test it
    :return: nothing
    """
    if TRAIN_MODE:
        print("Training mode.")
        print("Loading data...")
        if LOAD_DATA_FROM_FILE:
            # If spectrograms were already created, load them from file
            data_x = np.load(DATA_X_FILE)
            data_y = np.load(DATA_Y_FILE)
        else:
            data_x, data_y = load_data()
            # Transform to 3 channel image for VGG16
            # Array was of shape (x, y, z, 1), and it will be (x, y, z, 3) to be considered as an RGB image
            print("Transforming data to 3 channels image...")
            data_x = np.squeeze(np.stack((data_x,) * 3, -1))
            print("Done.")

        print("Data loaded.")

        if SAVE_DATA:
            # Saving spectrograms to avoid useless computation
            print("Saving data...")
            np.save(DATA_X_FILE, data_x)
            np.save(DATA_Y_FILE, data_y)
            print("Data saved.")

        # Shape of an input data of the CNN
        input_shape = data_x[0].shape

        print("Splitting data...")
        # Use of sklearn function because it uses less memory
        data_x, test_x, data_y, test_y = train_test_split(data_x, data_y, test_size=0.3, random_state=42)
        print("Done.")

        model = create_model(input_shape)
        log(model.summary())

        # Create model using Adam, an algorithm for stochastic optimization
        model.compile(loss="categorical_crossentropy",
                      optimizer='adam',
                      metrics=["accuracy"])

        # Train the model
        model_info = model.fit(data_x, data_y,
                               epochs=10,
                               batch_size=128,
                               verbose=1,
                               validation_data=(test_x, test_y))
        score = model.evaluate(test_x, test_y, verbose=0)
        print("Accuracy is {:.3f}".format(score[1]))

        # Save model
        model.save('./model.h5')
        print("Model saved.")

    else:
        print("Testing mode")
        # Load model
        model = load_model('./model.h5')

        # Load song
        signal, _ = librosa.load(TEST_FILE)

        # Process song
        signal = signal[:SAMPLE_SIZE]
        splits, _ = split_song(signal, 0)
        spectrograms = generate_spectrograms(splits)
        spectrograms = np.squeeze(np.stack((spectrograms,) * 3, -1))

        # Run into model
        results = model.predict(x=spectrograms, batch_size=len(spectrograms))

        # Interpret results
        genre = np.zeros(NUM_GENRES)
        for instance in results:
            genre[instance.argmax(axis=0)] += 1
        keys = list(GENRES.keys())
        values = list(GENRES.values())
        answer = keys[values.index(genre.argmax(axis=0))]
        print("The genre of the song is {0}".format(answer))


def create_test_data(data_x, data_y, test_size=0.3):
    """
    Split data to create test set by using a percentage of the total data
    Not used because of memory usage
    :param data_x: the x values of the data
    :param data_y: the y values of the data
    :param test_size: between 0 and 1 - the percentage of data used for test
    :return: the training set without the test instances, and the test set
    """
    test_x = []
    test_y = []
    for _ in range(int(len(data_x)*test_size)):
        r = random.randint(0, len(data_x)-1)
        x = data_x[r]
        y = data_y[r]
        test_x.append(x)
        test_y.append(y)
        data_x = np.delete(data_x, r, axis=0)
        data_y = np.delete(data_y, r, axis=0)
    return data_x, data_y, np.array(test_x), np.array(test_y)


def split_song(signal, genre, window_size=0.1, overlap_percent=0.5):
    """
    Split the signal in multiple overlapping windows
    :param signal: the time series of the sound
    :param genre: the genre of the sound
    :param window_size: the size of the window (percentage of the song,
        if song is 30s long, window_size of 0.1 will create windows of size 3s)
    :param overlap_percent: the percentage of overlap between windows (if window is 3s long, 50% of the
        next window will overlap with it)
    :return: the song splitted, and the corresponding genre
    """
    # Results
    t_x = []
    t_y = []

    # Shape like (x,)
    signal_length = signal.shape[0]
    # Size of a window
    window = int(signal_length * window_size)
    # Size of the offset
    offset = int(window * (1. - overlap_percent))
    # Split the signal
    # signal_length - window + offset is used because the last window will cover the last part
    parts = [signal[i:i + window] for i in range(0, signal_length - window + offset, offset)]
    for s in parts:
        t_x.append(s)
        t_y.append(genre)

    return np.array(t_x), np.array(t_y)


def generate_spectrograms(signals):
    """
    Generate the mel spectrogram of each signal of the given array
    :param signals: the list of time series
    :return: the list of mel spectrograms
    """
    # n_fft=1024, hop_length=512
    # n_fft=n_fft, hop_length=hop_length
    temp = []
    for instance in signals:
        # Create spectrograms and add a new axis (nb of channels = 1 for use in CNN as image)
        # Librosa create spectrogram like (time, frequency)
        # We convert it to (time, frequency, channel)
        temp.append(librosa.feature.melspectrogram(instance)[:, :, np.newaxis])
    return np.array(temp)


def load_data():
    """
    Load the audio files, split the songs and create spectrogram of each window
    :return: an array with x values, an array with y values
    """
    limit = 1
    count = 0
    log("Creating spectrograms...")
    data_x = []
    data_y = []
    # Walk the directory
    rootdir = os.getcwd()
    for subdir in os.listdir(rootdir + "\\" + DATA_FOLDER):
        path = rootdir + "\\" + DATA_FOLDER
        for file in os.listdir(path + "\\" + subdir):
            signal, sr = librosa.load(path + "\\" + subdir + "\\" + file)
            log("Processing file {0}".format(file))
            # Limit song size, to be sure it's 30s
            # 661500 is the number of values
            # 30s at 22050Hz makes 661500 values
            # But set a 660000 because after some tests, 661500 can cause problems
            signal = signal[:SAMPLE_SIZE]
            # The genre is in the name of the file, so we can use it
            genre = GENRES[file.split('.')[0]]
            # Split song
            splits, genres = split_song(signal=signal, genre=genre)
            spectrograms = generate_spectrograms(splits)

            data_y.extend(genres)
            data_x.extend(spectrograms)
            count += 1
            if count >= limit:
                break
    log("Done.")
    return np.array(data_x), np.array(to_categorical(data_y))


def log(msg):
    """
    Print a message if DEBUG is set to True
    :param msg: the message to log
    :return: nothing
    """
    if DEBUG:
        print(msg)


if __name__ == "__main__":
    main()
