#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/gdrive')


# In[ ]:


from keras.datasets import cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K

import matplotlib.pyplot as plt

import numpy as np


# # Loading Dataset

# First of all, we load CIFAR10 which is part of the Keras Datasets. 

# In[ ]:


(X_train, y_train), (_, _) = cifar10.load_data()


# ## data visualization

# lets take a look at some of the pictures in the dataset.

# In[ ]:


from mpl_toolkits.axes_grid1 import ImageGrid
fig = plt.figure(figsize=(10, 12))
grid = ImageGrid(fig, 111,
                 nrows_ncols=(3,4),
                 axes_pad=0.01,)

for i in range(12):
    grid[i].imshow(X_train[i])


# ## PreProccessing

# For data preprocessing phase, we changed type of numbers to float for speeding up training of model and normalize data for better results and faster training. 

# In[ ]:


# Rescale -1 to 1
X_train = (X_train / 255 - 0.5) * 2
X_train = np.clip(X_train, -1, 1)
# X_train = np.expand_dims(X_train, axis=3)
y_train = y_train.reshape(-1, 1)

num_classes = 10

labels = to_categorical(y_train, num_classes=num_classes+1)


# #Generator

# In[ ]:


num_classes = 10
noise_dim = 100


# The Generator mainly consists of convoloutional layers with ReLU activation function with kernel size 3 and Batch Normalization on every layer and the activation function of the last layer is tanh.
# 
# Upsampling was also used to create image with desired shape from the noise input vector.

# In[ ]:


generator = Sequential()

generator.add(Dense(128 * 8 * 8, activation="relu", input_dim=noise_dim))
generator.add(Reshape((8, 8, 128)))
generator.add(BatchNormalization(momentum=0.8))
generator.add(UpSampling2D())
generator.add(Conv2D(128, kernel_size=3, padding="same"))
generator.add(Activation("relu"))
generator.add(BatchNormalization(momentum=0.8))
generator.add(UpSampling2D())
generator.add(Conv2D(64, kernel_size=3, padding="same"))
generator.add(Activation("relu"))
generator.add(BatchNormalization(momentum=0.8))
generator.add(Conv2D(3, kernel_size=3, padding="same"))
generator.add(Activation("tanh"))

generator.summary()


# #Discriminator

# Then we define the Discriminator part.
# 
# It's pretty much like the Generator part but in the last layer it has two outputs, one showing the validity of input and the other showing the class it belongs to.
# 
# And for the activation function, LeakyReLu is used.

# In[ ]:


img_shape = X_train[0].shape
        
model = Sequential()

model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
model.add(LeakyReLU(alpha=0.2))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
model.add(ZeroPadding2D(padding=((0,1),(0,1))))
model.add(LeakyReLU(alpha=0.2))
model.add(Dropout(0.25))
model.add(BatchNormalization(momentum=0.8))
model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
model.add(LeakyReLU(alpha=0.2))
model.add(Dropout(0.25))
model.add(BatchNormalization(momentum=0.8))
model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
model.add(LeakyReLU(alpha=0.2))
model.add(Dropout(0.25))

model.add(Flatten())

img = Input(shape=img_shape)

features = model(img)
validity = Dense(1, activation="sigmoid")(features)
label = Dense(num_classes+1, activation="softmax")(features)

discriminator =  Model(img, [validity, label])

discriminator.summary()


# #SGAN

# Now we compile both models above using binary cross entropy and Adam optimizer.
# 
# And finally it's time to build the whole SGAN.

# In[ ]:


from keras.optimizers import Adam
noise_shape = (noise_dim, )

discriminator.compile(Adam(lr=0.0003, beta_1=0.5), loss='binary_crossentropy',
                      metrics=['binary_accuracy'])


discriminator.trainable = False

z = Input(shape=noise_shape)
img = generator(z)
validity, _ = discriminator(img)
sgan = Model(inputs=z, outputs=validity)

sgan.compile(Adam(lr=0.0004, beta_1=0.5), loss='binary_crossentropy',
            metrics=['binary_accuracy'])

sgan.summary()


# #Training

# Number of Epochs is 100.
# 
# Batch size is 32.

# In[ ]:


batch_size = 32
half_batch = batch_size / 2
epochs = 100


# Now we define a function that generates a n dimentional normal noise.

# In[ ]:


def gennoise(batch_size, noise_dim=100): 
    x = np.random.normal(0, 1.0, (batch_size, noise_dim))
    return x


# For each epoch, at first we train discriminator with real and fake samples and then train the GAN while freezing the weights of discriminator part.

# In[ ]:


valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

d_loss = []
g_loss = []

for e in range(epochs + 1):
    for i in range(len(X_train) // batch_size):
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        discriminator.trainable = True
        
        # Real samples
        x_batch = X_train[i*batch_size:(i+1)*batch_size]
        y_batch = labels[i*batch_size:(i+1)*batch_size]
        d_loss_real = discriminator.train_on_batch(x=x_batch, y=[valid, y_batch])
        
        # Fake Samples
        z = gennoise(batch_size, noise_dim)
        x_fake = generator.predict_on_batch(z)
        y_fake = to_categorical([num_classes for i in range(batch_size)], num_classes=num_classes+1)
        d_loss_fake = discriminator.train_on_batch(x=x_fake, y=[fake, y_fake])
         
        # Discriminator loss
        d_loss_batch = 0.5 * (d_loss_real[0] + d_loss_fake[0])
        
        # ---------------------
        # Train Generator
        # ---------------------
        discriminator.trainable = False
        g_loss_batch = sgan.train_on_batch(x=z, y=valid)

    
    d_loss.append(d_loss_batch)
    g_loss.append(g_loss_batch[0])
    print('epoch = %d/%d, d_loss=%.3f, g_loss=%.3f' % (e + 1, epochs, d_loss[-1], g_loss[-1]), 100*' ')


    if e % 20 == 0:
        fig = plt.figure(figsize=(10, 12))
        grid = ImageGrid(fig, 111,
                        nrows_ncols=(10,10),
                        axes_pad=0.01,)
        for i in range(100):
            grid[i].imshow((((generator.predict(gennoise(1))) + 1)* 127).reshape((32,32, 3)).astype(np.uint8))
        plt.show()
        name = 'pic' + str(e) + '.png'
        fig.savefig('gdrive/My Drive/NN/MiniProj3/Result 3-2/'+name)


# And finally the Loss figures.

# In[ ]:


plt.plot(d_loss)
plt.plot(g_loss)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Discriminator', 'Generator'])
plt.show()


# In[ ]:




