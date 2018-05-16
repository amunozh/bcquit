import audio_dataset
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import optimizers
import numpy as np

def classifier_model(): #Building of the CNN
    model = models.Sequential()

    model.add(layers.Conv2D(1, [2,10], input_shape=(40, 44,1), strides=(1,1), padding='valid', activation='relu',data_format='channels_last'))

    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))

    model.add(layers.Conv2D(1, [2, 6], strides=(1, 1), padding='valid', activation='relu'))

    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))

    model.add(layers.Conv2D(1, [2, 3], strides=(1, 1), padding='valid', activation='relu'))

    model.add(layers.Flatten())

    model.add(layers.Dense(1))

    print(model.summary())
    return model



if __name__== "__main__":

    data = audio_dataset.read_files()
    f = []
    for i in range(0, data.shape[0]):
        audio = data[i, 0:-2]
        f.append(audio_dataset.features(audio, 44100))

    l=173
    f=np.array(f)
    print(f.shape)
    X_train = np.reshape(f,[f.shape[0],40,44,1])
    Y_train = data[:,-1]

    np.random.seed(100)
    My_model=classifier_model()
    sgd = optimizers.SGD(lr=0.1, clipnorm=1.)
    My_model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    My_model.fit(X_train, Y_train,  batch_size=4, nb_epoch=200, verbose=1)



