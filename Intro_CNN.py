from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) \
    = mnist.load_data()

# Take a look on mnist

train_labels[0:10]

import matplotlib.pyplot as plt

digit = train_images[0] # Try 2 and 9
plt.imshow(digit, cmap = plt.cm.binary)
plt.show()

# Scale and Labels
# Reshape the image data to a vector
train_images_ = train_images.reshape((60000, 28 * 28))
train_images_ = train_images_.astype('float32') / 255

test_images_ = test_images.reshape((10000, 28 * 28))
test_images_ = test_images_.astype('float32') / 255

from keras.utils import to_categorical

train_labels_ = to_categorical(train_labels)
test_labels_ = to_categorical(test_labels)

# Model

from keras import models
from keras import layers

#three layer feed forward dense NN
network = models.Sequential()
network.add(layers.Dense(512, activation = 'relu', 
                         input_shape = (28 * 28, )))
#(28*28) +1 parameter for one neuron -> total: (28*28 + 1) * 512 = 401920 parameters
network.add(layers.Dense(10, activation = 'softmax'))

network.summary()

# Besides relu, we may also try sigmoid, tanh, or others.

network.compile(optimizer = 'rmsprop', 
                loss = 'categorical_crossentropy',
                metrics = ['accuracy'])

# Besides rmsprop, we may also try SGD(stochastic GD), Adam, or others.

# Train

network.fit(train_images_, train_labels_, 
            epochs = 5, batch_size = 128)

# We may also try # of epochs other than 5.

# Test

train_loss, train_acc = \
    network.evaluate(train_images_, train_labels_)
print('train_acc:', train_acc)

test_loss, test_acc = \
    network.evaluate(test_images_, test_labels_)
print('test_acc:', test_acc)

prediction_labels_ = network.predict(test_images_)

#check the model wrong prediction
import numpy as np
np.sum(prediction_labels_[0])

np.argmax(prediction_labels_[0])
prediction_labels = np.argmax(prediction_labels_, axis = 1)
#預測錯誤的index
index = np.nonzero(test_labels - prediction_labels)

index
test_labels[index]
prediction_labels[index]
#show the graph which we wrongly predicted
digit = test_images[321]
plt.imshow(digit, cmap = plt.cm.binary)
plt.show()

#from keras.utils.vis_utils import plot_model
#plot_model(network, to_file = 'network_plot.png',
#           show_shapes = True,
#           show_layer_names = True)

# Convulutional Nerual Networks
# a.k.a., CNN, or convnet

# We use the same data. However, the shape is different.

train_images_ = train_images.reshape((60000, 28, 28, 1))
train_images_ = train_images_.astype('float32') / 255

test_images_ = test_images.reshape((10000, 28, 28, 1))
test_images_ = test_images_.astype('float32') / 255

# Model

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', 
                          input_shape = (28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(10, activation = 'softmax'))

model.summary()

model.compile(optimizer = 'rmsprop', 
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

# Besides rmsprop, we may also try SGD, Adam, or others.

# Train

model.fit(train_images_, train_labels_, 
          epochs = 5, batch_size = 64)

test_loss, test_acc = \
    model.evaluate(test_images_, test_labels_)
print('test_acc:', test_acc)