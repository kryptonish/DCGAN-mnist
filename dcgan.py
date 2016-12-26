import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.layers import Input, Convolution2D, Dropout, Flatten, Dense, BatchNormalization, Reshape, UpSampling2D
from keras import backend as K
import matplotlib.pyplot as plt
import time
import math
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, sgd
from keras.layers.noise import GaussianNoise
from PIL import Image

#This code aims to construct a DCGAN that generates MNIST digits.
#First, I code up a discriminator which is a standard 2D convolutional neural network.


num_filters = 48
filter_size = [3,3]
gen_conv_filter = 56
batch_size = 60
trainer = 1000
update_weight_val = 100

def normalize_data(data):
	return (data.astype(np.float32) - 127.5)/127.5


def denormalize_data(data):
    return data*127.5+127.5



def discriminator():
	disc = Sequential()
	disc.add(Convolution2D(num_filters, filter_size[0], filter_size[1], border_mode='same',input_shape=(28,28,1)))
	disc.add(Activation('relu'))
	disc.add(Convolution2D(num_filters, filter_size[0], filter_size[1], border_mode='same'))
	disc.add(Activation('relu'))
	disc.add(Dropout(0.2))
	disc.add(MaxPooling2D(pool_size=(2, 2)))
	disc.add(Flatten())
	disc.add(Dense(128))
	disc.add(Activation('relu'))
	disc.add(Dropout(0.2))
	disc.add(Dense(1)) #only one output
	disc.add(Activation('sigmoid'))
	return disc
# print(discriminator.summary())

def generator():
	gen = Sequential()
	gen.add(Dense(input_dim=100, output_dim=1024))
	# gen.add(GaussianNoise(3))

	gen.add(Activation('relu'))
	gen.add(Dense(256*7*7))
	# gen.add(Activation('LeakyReLU'))
	gen.add(Activation('relu'))
	gen.add(Dropout(0.2))
	gen.add(Reshape((7, 7, 256)))
	gen.add(UpSampling2D(size=(2, 2)))
	gen.add(Convolution2D(128, 5, 5, border_mode='same'))
	gen.add(Activation('relu'))
	gen.add(GaussianNoise(3))
	gen.add(Dropout(0.2))
	gen.add(UpSampling2D(size=(2, 2)))
	gen.add(Convolution2D(1, 5, 5, border_mode='same'))
	gen.add(Activation('relu'))
	return gen



def gan(gen,disc):
	model = Sequential()
	model.add(gen)
	disc.trainable = False
	model.add(disc)
	return model

def gan_train():
	print "batch_size : ", batch_size, "trainer : ",trainer
	(train_x, train_y), (test_x, test_y) = mnist.load_data()
	train_x = train_x.reshape(len(train_x), 28, 28, 1).astype('float32')
	# train_x = normalize_data(train_x)
	# train_x -= 128.0
	# train_x /= 255.0

	train_y_gan = np.random.uniform(0.7,1.2,len(train_y))
	test_y_gan = np.random.uniform(0.7,1.2,len(train_y))

	Generator = generator()
	Discriminator = discriminator()
	GAN = gan(Generator,Discriminator)
	discriminator_optim = sgd(lr=0.008)


	GAN.compile(loss='binary_crossentropy',optimizer='SGD',metrics=['accuracy'])
	Generator.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
	Discriminator.compile(loss='binary_crossentropy',optimizer=discriminator_optim,metrics=['accuracy'])

	for i in range(trainer):
		generator_input = np.random.normal(2, 2, size = (batch_size,100))
		generator_output = Generator.predict(generator_input)
		generator_train_labels = train_y_gan
		generator_test_labels = test_y_gan

		discriminator_train_input = train_x[np.random.randint(0,len(train_x),batch_size)]
		discriminator_input = np.concatenate((generator_output,discriminator_train_input))
		discriminator_labels = np.concatenate(( np.random.uniform(0,0.3,batch_size) , np.random.uniform(0.7,1.2,batch_size) ))
		a = np.random.randint(0,len(discriminator_labels),batch_size/3)
		for val in a:
			if val%2 == 0:
				discriminator_labels[a] = 0.2
			else:
				discriminator_labels[a] = 1.0


		discriminator_loss = Discriminator.train_on_batch(discriminator_input, discriminator_labels)
		Discriminator.trainable = False

		gan_labels = np.random.uniform(0.7,1.2,batch_size)
		gan_input = np.random.normal(3, 2, size = (batch_size,100))
		gan_loss = GAN.train_on_batch(gan_input, gan_labels)
		Discriminator.trainable = True
		print discriminator_loss,gan_loss
		if i % 100 == 0:
			print "SAVING, trainer id : " , i
			Generator.save_weights('generator'+str(i), True)
			Discriminator.save_weights('discriminator'+str(i), True)
	Generator.save_weights('generator', True)
	Discriminator.save_weights('discriminator', True)


def generate():
	Generator = generator()
	Generator.compile(loss='binary_crossentropy', optimizer="adam")
	Generator.load_weights('generator')
	Discriminator = discriminator()
	Discriminator.compile(loss='binary_crossentropy', optimizer="SGD")
	Discriminator.load_weights('discriminator')
	generator_input = np.random.normal(1, 2, size = (batch_size,100))
	generator_output = Generator.predict(generator_input)
	# generator_output = denormalize_data(generator_output)
	inn = generator_output[:,:,:,0]
	for val in range(len(inn)):
		im = Image.fromarray(inn[val].astype('float32'))
		im.convert('LA').save("your_file"+str(val)+".png")

	print np.shape(inn)





gan_train()
generate()
# generator()