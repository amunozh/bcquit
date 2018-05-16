import librosa
import numpy as np
import os
import glob
from librosa import display as disp
import matplotlib.pyplot as plt


def read_files():
    '''this program reads the .wave files in the 4 folders corresponding to each class, and creates
     a matrix with dimensions [#files, th+5], where #file is the total number of .wav files in the 4 folders,
     th is the size of the smallest file, and the last 5 columns correspond to binary values that specify the clasess
     with the following order:
         female=0/male=1, hu, lo, sc, dk'''

    # Load sound file
    path = '/home/andres/Escritorio/CNN'  # --->introduce the location of the 4 folders

    numfiles = 61  # -->if you want to introduce the #files manually
    th = 22050
    splitname = []
    matrix = np.zeros((numfiles*5, th + 2))
    # %%
    ii = 0
    classes = ['/hu', '/nothu']
    for f in classes:
        folder = path + f
        for filename in glob.glob(os.path.join(folder, '*.wav')):  # specify the directory
            # access     the information of the wav file:  name->filename   content->y
            y, sr = librosa.load(filename)
            for seconds in range(1,5):
                crop = y[int(seconds * sr):int((seconds+1) * sr)]
                splitname = filename.split("-")
                matrix[ii, 0:th] = crop
                # specify the features at the end of the row
                if splitname[7] == 'm':
                    matrix[ii, th] = 1
                if f == '/hu':
                    matrix[ii, th + 1] = 1
                ii += 1

    return matrix


def features(y,sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    log_S = librosa.power_to_db(mfccs)
    return(log_S)


if __name__ == '__main__':
    data=read_files()
    f=[]
    for i in range(0,data.shape[0] - 1):
        audio=data[i , 0:-2]
        f.append(features(audio,44100))


