import audio_dataset
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import initializers
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def classifier_model(): #Building of the CNN
    model = models.Sequential()

    model.add(layers.Conv2D(1, [2,40], input_shape=(1, 40, 173), strides=(1,1), padding='valid', activation='relu'))
    #
    #model.add(layers.MaxPool1D(pool_size=2, strides=2, padding='valid'))
    #
    model.add(layers.Conv2D(1, [2, 20], strides=(1, 1), padding='valid', activation='relu',
                             kernel_initializer=initializers.glorot_normal(), bias_initializer=initializers.Zeros()))
    #
    # model.add(layers.MaxPool1D(pool_size=2, strides=2, padding='valid'))
    #
    #
    model.add(layers.Conv2D(1, [2,10], strides=(3, 3), padding='valid', activation='relu',
                              kernel_initializer=initializers.glorot_normal(), bias_initializer=initializers.Zeros()))
    #
    # model.add(layers.MaxPool1D(pool_size=2, strides=2, padding='valid'))
    #
    model.add(layers.Flatten())
    #
    model.add(layers.Dense(1, kernel_initializer=initializers.glorot_normal(),
                            bias_initializer=initializers.Zeros()))

    print(model.summary())
    return model



def training(length, l_coeff, train_steps, data, results):

    # define placeholders for batch of training images and labels
    x = tf.placeholder(tf.float32, shape=(None, 40, 173,1), name='x') #input
    y = tf.placeholder(tf.float32, shape=(None , 1), name='y') #output

    # create model
    my_net = classifier_model()

    # use model on input image batch to compute
    h = my_net(x)

    # define loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=h))

    # define the optimizer and what is optimizing
    optimizer = tf.train.GradientDescentOptimizer(l_coeff)
    train_step = optimizer.minimize(loss)

    # create session
    sess = tf.InteractiveSession()

    # initialize variables
    tf.global_variables_initializer().run()

    costs=[]
    for i in range(train_steps):
        # generate the data
        xval = data
        yval = results
        # train
        train_data = {x: xval, y: yval}  # define the train data
        sess.run(train_step, feed_dict = train_data)  # run the session with the neural network created and with that train data

        if i % 100 == 0:  # print the intermediate result
            costs.append(loss.eval(feed_dict=train_data,session=sess))

    plt.plot(costs)

if __name__== "__main__":
    data = audio_dataset.read_files()
    f = []
    for i in range(0, data.shape[0] - 1):
        audio = data[i, 0:-2]
        f.append(audio_dataset.features(audio, 44100))

    l=173
    f=np.array(f)
    f=np.reshape(f,[f.shape[0],1,40,173])
    training(l, 0.01, 1000, f, data[:, -1])

