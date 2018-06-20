from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
import numpy as np

test_image = np.random.randn(224,224,3)
#test_image = np.expand_dims(test_image,axis=0)
#print(test_image.shape)
"""
print(test_image)
plt.imshow(test_image)
plt.show()
"""

model=Sequential()

model.add(Conv2D(96,kernel_size=(11,11),padding='valid',strides=4,input_shape=test_image.shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

#(1, 26, 26, 96)


exp_test = np.expand_dims(test_image,axis=0)
conv=model.predict(exp_test)
print(conv.shape)
#conv=model.predict(test_image)
#print(conv.shape)
