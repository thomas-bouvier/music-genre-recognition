import librosa
import numpy as np
import os
from keras.utils import to_categorical
import random
from model import create_model
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import argparse
import itertools

GENRES = {'metal': 0, 'disco': 1, 'classical': 2, 'hiphop': 3, 'jazz': 4,
          'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'rock': 9}
NUM_GENRES = len(GENRES)
DATA_FOLDER = "./musics"
DATA_Y_FILE = './data_y.npy'
DATA_X_FILE = './data_x.npy'
MODEL_FILE = './model.h5'
BATCH_SIZE = 1000
# Set to 660000 instead of 661500 because depending on the format, it can cause problems
SAMPLE_SIZE = 660000

ACCEPTED_MODES = ['train', 'test']


def main(args):
    """
    Main function of the program
    If training mode, load the data, train the model and save it
    If testing mode, load the music, load the model and test it
    :return: nothing
    """
    # Check arguments
    # Mode
    if args.mode.lower() is not None:
        if args.mode in ACCEPTED_MODES:
            mode = args.mode
        else:
            raise ValueError('Invalid mode parameter.')
    else:
        mode = 'test'

    # Check if data files exists
    if args.load_data:
        if not os.path.isfile(DATA_X_FILE) or not os.path.isfile(DATA_Y_FILE):
            print("Files containing data do not exists, so they will be created.")
            args.load_data = False

    # Load and save data
    if args.load_data and args.save_data:
        print("Since load data flag has been set, data won't be saved.")
        args.save_data = False

    # Test song
    if mode == 'test':
        if args.song is not None:
            if os.path.exists("./{0}".format(args.song)):
                test_song = "./{0}".format(args.song)
            else:
                raise ValueError("The specified test song doesn't exist or is not in the root folder.")
        elif args.folder is not None:
            if os.path.exists("./{0}".format(args.folder)):
                test_folder = "./{0}".format(args.folder)
            else:
                raise ValueError("The specified folder doesn't exist or is not in the root folder.")
        else:
            raise ValueError("No song or folder were given for testing.")

    if mode == 'train':
        print("Training mode.")
        print("Loading data...")
        if args.load_data:
            # If spectrograms were already created, load them from file
            data_x = np.load(DATA_X_FILE)
            data_y = np.load(DATA_Y_FILE)
        else:
            data_x, data_y = load_data(args.debug)

        print("Data loaded.")

        if args.save_data:
            # Saving spectrograms to avoid useless computation
            print("Saving data...")
            np.save(DATA_X_FILE, data_x)
            np.save(DATA_Y_FILE, data_y)
            print("Data saved.")

        # Shape of an input data of the CNN
        input_shape = data_x[0].shape

        print("Splitting data...")
        # Use of sklearn function because it uses less memory
        data_x, test_x, data_y, test_y = train_test_split(data_x, data_y, test_size=0.3, random_state=666)
        print("Done.")

        model = create_model(input_shape, NUM_GENRES)

        if args.debug:
            print(model.summary())

        # Create model using Adam, an algorithm for stochastic optimization
        # and categorical cross entropy as loss function
        model.compile(loss="categorical_crossentropy",
                      optimizer='adam',
                      metrics=["accuracy"])

        # Train the model
        model_info = model.fit(data_x, data_y,
                               epochs=50,
                               batch_size=BATCH_SIZE,
                               verbose=1,
                               validation_data=(test_x, test_y))
        score = model.evaluate(test_x, test_y, verbose=0)
        print("Accuracy is {:.3f}".format(score[1]))

        # Save model
        if args.save_model:
            print("Saving model...")
            model.save(MODEL_FILE)
            print("Model saved.")

        # Show graphs of accuracy and loss over time
        plt.figure(figsize=(15, 7))

        plt.subplot(1, 2, 1)
        plt.plot(model_info.history['acc'], label='train')
        plt.plot(model_info.history['val_acc'], label='validation')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(model_info.history['loss'], label='train')
        plt.plot(model_info.history['val_loss'], label='validation')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

        classes = ['metal', 'disco', 'classical', 'hiphop', 'jazz', 'country', 'pop', 'blues', 'reggae', 'rock']
        y_pred = model.predict(test_x)
        conf_matrix = confusion_matrix(y_true=np.argmax(test_y, axis=1),
                                       y_pred=np.argmax(y_pred, axis=1))
        plot_confusion_matrix(conf_matrix, classes=classes, title='Confusion matrix', normalize=True)
        plt.show()

    else:
        print("Testing mode")
        # Load model
        model = load_model(MODEL_FILE)

        if args.debug:
            print(model.summary())

        if args.song is not None:
            # Load song
            signal, _ = librosa.load('./' + test_song)

            # Process song
            nb_samples = int(len(signal)/SAMPLE_SIZE)
            spectrograms = []
            for i in range(nb_samples):
                part = signal[i * SAMPLE_SIZE: (i+1) * SAMPLE_SIZE]
                splits, _ = split_song(part, 0)
                spectr_part = generate_spectrograms(splits)

                # Uncomment this part if you want to use 2D Convolutions
                # It will transform the grayscale spectrogram to a RGB image
                # spectr_part = np.squeeze(np.stack((spectr_part,) * 3, -1))

                spectrograms.extend(spectr_part)

            spectrograms = np.array(spectrograms)

            # Run into model
            results = model.predict_classes(x=spectrograms)
            # results = model.predict(x=spectrograms, batch_size=len(spectrograms))
            if args.debug:
                print(results)

            # Interpret results
            genre = np.zeros(NUM_GENRES)
            keys = list(GENRES.keys())
            values = list(GENRES.values())
            for instance in results:
                genre[instance] += 1

            print("The genre of the song is:")
            for i in range(NUM_GENRES):
                print("{0} ".format(keys[values.index(i)].title()) +
                      "at {:.3f} %".format(genre[i] * 100 / sum([x for x in genre])))

        elif args.folder is not None:
            # Folder containing songs
            rootdir = os.getcwd()
            for file in os.listdir(rootdir + "/" + test_folder):
                path = rootdir + "/" + test_folder
                if os.path.isfile(path + "/" + file):
                    spectrograms = []
                    signal, sr = librosa.load(path + "/" + file)
                    if args.debug:
                        print("\nProcessing file {0}".format(file))
                    nb_samples = int(len(signal) / SAMPLE_SIZE)
                    for i in range(nb_samples):
                        part = signal[i * SAMPLE_SIZE: (i + 1) * SAMPLE_SIZE]
                        splits, _ = split_song(part, 0)
                        spectr_part = generate_spectrograms(splits)

                        # Uncomment this part if you want to use 2D Convolutions
                        # It will transform the grayscale spectrogram to a RGB image
                        # spectr_part = np.squeeze(np.stack((spectr_part,) * 3, -1))

                        spectrograms.extend(spectr_part)

                    spectrograms = np.array(spectrograms)

                    # Run into model
                    results = model.predict_classes(x=spectrograms)

                    # Interpret results
                    genre = np.zeros(NUM_GENRES)
                    keys = list(GENRES.keys())
                    values = list(GENRES.values())
                    for instance in results:
                        genre[instance] += 1

                    print("The genre of the song {0} is:".format(file))
                    for i in range(NUM_GENRES):
                        print("    {0} ".format(keys[values.index(i)].title()) +
                              "at {:.3f} %".format(genre[i] * 100 / sum([x for x in genre])))


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
        # Random index
        r = random.randint(0, len(data_x)-1)
        # Get value
        x = data_x[r]
        y = data_y[r]
        # Add it to test set
        test_x.append(x)
        test_y.append(y)
        # Remove it from training set
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
    x = []
    y = []
    # Shape like (x,), so we get x
    signal_length = signal.shape[0]
    # Size of a window
    size_part = int(signal_length * window_size)
    # Size of the offset
    offset = int(size_part * overlap_percent)
    # Limit of range
    limit = signal_length - size_part + offset
    # Split the signal
    for i in range(0, limit, offset):
        # Add the part of the signal to the array
        x.append(signal[i:i+size_part])
        y.append(genre)

    return np.array(x), np.array(y)


def generate_spectrograms(signals, conv_1d=True):
    """
    Generate the mel spectrogram of each signal of the given array
    :param signals: the list of time series
    :param conv_1d: set to false if 2d convolutions will be used, it will add a new axis
    :return: the list of mel spectrograms
    """
    rep = []
    for instance in signals:
        # Create spectrograms and add a new axis (nb of channels = 1 for use in CNN as image)
        # Librosa create spectrogram like (time, frequency)
        # We convert it to (time, frequency, channel)
        # But if we use 1D convolutional layer, no need to add axis
        if conv_1d:
            rep.append(librosa.feature.melspectrogram(instance))
        else:
            rep.append(np.expand_dims(librosa.feature.melspectrogram(instance), axis=2))
    return np.array(rep)


def load_data(debug):
    """
    Load the audio files, split the songs and create spectrogram of each window
    :return: an array with x values, an array with y values
    """
    if debug:
        print("Creating spectrograms...")
        
    data_x = []
    data_y = []
    # Walk the directory
    rootdir = os.getcwd()
    for subdir in os.listdir(rootdir + "/" + DATA_FOLDER):
        path = rootdir + "/" + DATA_FOLDER
        if os.path.isdir(path + "/" + subdir):
            for file in os.listdir(path + "/" + subdir):
                if os.path.isfile(path + "/" + subdir + "/" + file):
                    signal, sr = librosa.load(path + "/" + subdir + "/" + file)
                    if debug:
                        print("Processing file {0}".format(file))
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
            
    if debug:
        print("Done.")
    
    return np.array(data_x), np.array(to_categorical(data_y))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Provided by Sklearn documentation.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "-mode", dest="mode",
                        help="Chosen mode of the program. Can be train or test. By default it's test.", metavar="MODE")
    parser.add_argument("--no-save-data", dest="save_data",
                        action='store_false', default=True,
                        help="Data files are saved by default, this tells the program to not save them.")
    parser.add_argument("--no-save-model", dest="save_model",
                        action='store_false', default=True,
                        help="Models are saved by default, this tells the program to not save them.")
    parser.add_argument("--load-data", dest="load_data",
                        default=False, action='store_true',
                        help="Load data from the .npy files, by default they are not loaded.")
    parser.add_argument("-song", dest="song",
                        help="The name of the song file used for test.")
    parser.add_argument("-folder", dest="folder",
                        help="The name of the folder containing the song files used for test.")
    parser.add_argument("--debug", dest="debug",
                        action='store_true', default=False,
                        help="Enable debug mode.")
    args = parser.parse_args()
    main(args)
