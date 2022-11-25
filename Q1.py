#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import cv2 
from PIL import Image 
import matplotlib.pyplot as plt
import keras
from keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape
from keras.layers import BatchNormalization
from keras.models import Model
from keras.datasets import mnist
from keras.losses import binary_crossentropy
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt


# # Load MNIST Dataset and Data Preprocessing

# First of all, we load MNIST which is part of the Keras Datasets. For data preprocessing phase, we changed type of numbers to float for speeding up training of model and normalize data for better results and faster training. Then we resized data so that each sample takes (28, 28, 1) shape.

# In[ ]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = np.array([i / 255 for i in x_train])
x_test = np.array([i / 255 for i in x_test])

x_train.resize((60000, 28, 28, 1))
x_test.resize((10000, 28, 28, 1))

x_train.shape, x_test.shape, y_train.shape, y_test.shape


# # Encoder

# We define our encoder with one 2d Convolutional layer, one Dense layer and then two Dense layers named mu and sigma, which output mean and standard deviation, respectively.

# In[ ]:


latent_dim = 2
int_dim = 100

img_width, img_height = x_train[0].shape[0:2]


# In[ ]:


e_input = Input(shape=(28, 28, 1) , name='encoder_input')
x = Conv2D(20, kernel_size=(5, 5), padding="same",  name='conv')(e_input)
x = Flatten(name='flatten')(x)
x = Dense(int_dim, activation='relu', name='dense_encoder')(x)
mu = Dense(latent_dim, name='mean')(x)
sigma = Dense(latent_dim, name='variance')(x)


# ## Reparameterization

# In[ ]:


def sampler(args):

  mu, sigma = args
  
  batch = K.shape(mu)[0]
  dim_latent = mu.shape[1]
  
  eps = K.random_normal(shape=(batch, dim_latent))
  
  return mu + K.exp(sigma / 2) * eps


# When using Gradient Descent for optimizing our VAE, we are optimizing a loss function which requires to be differentiable. We reparameterized sample fed to the loss function into the shape 𝜇+𝜎^2×𝜖 so it now has the properties we needed to use Gradient Descent accurately.

# In[ ]:


z = Lambda(sampler, output_shape=(latent_dim, ), name='sampler')([mu, sigma])


# Now with this lambda layer applying sampling function given to it, gradients are computed correctly in the backward pass.

# In[ ]:


encoder = Model(e_input, [mu, sigma, z], name='encoder')
encoder.summary()


# We instantiated the encoder which takes inputs and outputs means, standard deviations and z (points sampled using sampler function using means and  standard deviations as inputs).

# # Decoder

# The decoder has two Dense layers and then a 2d Convolution Transpose layer. As you can see, the layers are in reverse order of the encoder.

# In[ ]:


d_input = Input(shape=(latent_dim, ), name='decoder_input')
x = Dense(int_dim, activation='relu', name='dense_decoder_1')(d_input)
x = Dense(img_height * img_width , activation='relu', name='dense_decoder_2')(x)
x = Reshape((img_height, img_width, 1), name='reshape_decoder')(x)
x = Conv2D(20, kernel_size=(5,5), padding="same",  name='conv')(x)
out = Dense(1, activation='relu', name='output')(x)


# In[ ]:


decoder = Model(d_input, out, name='decoder')
decoder.summary()


# # VAE

# The whole VAE consists of the encoder and then the decoder. We instantiated a VAE now that we have both encoder and decoder.

# In[ ]:


vae = Model(e_input, decoder(encoder(e_input)[2]), name='vae')
vae.summary()


# ## Reconstruction Loss

# For loss function we use mean of binary crossentropy loss and kl divergance loss. 

# In[ ]:


def kl_reconstruction_loss(true, pred):
  # Reconstruction loss
  reconstruction_loss = binary_crossentropy(K.flatten(true), K.flatten(pred)) * ( img_height * img_width)
  
  # KL divergence loss
  kl_loss = -1 - sigma + K.square(mu) + K.exp(sigma)
  kl_loss = 0.5 * K.sum(kl_loss, axis=-1)

  
  # Total loss = 50% rec + 50% KL divergence loss
  return 0.5 * reconstruction_loss + 0.5 * kl_loss


# # Plotting Functions

# In[ ]:


import random
from mpl_toolkits.axes_grid1 import ImageGrid

def show_sample(vae):
    fig = plt.figure(figsize=(20, 10))
    grid = ImageGrid(fig, 111,
                 nrows_ncols=(10,10),
                 axes_pad=0.01,)

    for _ in range(100):

        xi = random.uniform(-4, 4)
        yi = random.uniform(-4, 4)
        x_decoded = decoder.predict(np.array([[xi, yi]]))
        digit = x_decoded[0].reshape(img_width, img_height)
        grid[_].imshow(digit)
    


# In[ ]:


def show_train_loss_history(trained, title=None, show_validation=True):
    x1 = []
    x2 = []
    for i in trained:
        for j in i.history['loss']:
            x1.append(j)
        for j in i.history['val_loss']:
            x2.append(j)
    
    plt.plot(x1)
    if(show_validation):
      plt.plot(x2)      
      plt.legend(['train_loss', 'validation_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    if(title):
      plt.title(title)
    plt.show()


# # Compile and Train

# Batch size is 128, number of epochs is 100 and we use 20% of data for validation in each epoch.

# In[ ]:


batch_size = 128
n_epochs = 100
validation_split = 0.2

vae.compile(optimizer='adam', loss=kl_reconstruction_loss)


# Now we train our VAE on MNIST dataset. 100 epochs are applied in 5 rounds each having 20 epochs.

# In[ ]:


trained_vae = []
show_sample(vae)
plt.show()
for _ in range(4):
    trained_vae.append(vae.fit(x_train, x_train, epochs = n_epochs//4, batch_size = batch_size, validation_split = validation_split, shuffle=True))
    show_sample(vae)
    plt.show()


# ## train and validation loss

# Now we plot train and validation loss during training phase of out VAE model.

# In[ ]:


show_train_loss_history(trained_vae, "Training Loss Plot")


# # Visualization

# In[ ]:


def cluster_latent(encoder, data):
  input_data, target_data = data
  mu = encoder.predict(input_data)[0]
  plt.figure(figsize=(8, 10))
  plt.scatter(mu[:, 0], mu[:, 1], c=target_data)
  plt.xlabel('latent_dim 1')
  plt.ylabel('latent_dim 2')
  plt.colorbar()
  plt.show()


data = (x_test, y_test)
cluster_latent(encoder, data)


# Clusters constructed after applying encoder to input data.

# In[ ]:


def decode_latent(decoder, data, num_channels=1):
  num_samples = 15
  figure = np.zeros((img_width * num_samples, img_height * num_samples, num_channels))
  grid_x = np.linspace(-4, 4, num_samples)
  grid_y = np.linspace(-4, 4, num_samples)[::-1]


  for i, yi in enumerate(grid_y):
      for j, xi in enumerate(grid_x):
          x_decoded = decoder.predict(np.array([[xi, yi]]))
          digit = x_decoded[0].reshape(img_width, img_height, num_channels)
          figure[i * img_width: (i + 1) * img_width,
                  j * img_height: (j + 1) * img_height] = digit

  plt.figure(figsize=(10, 10))
  plt.title('decoder latent')
  
  figure = figure.reshape((np.shape(figure)[0], np.shape(figure)[1]))
  plt.imshow(figure)
  
decode_latent(decoder, data)
plt.show()


# In[ ]:




