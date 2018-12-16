
# coding: utf-8

# In[98]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Model
import random


# In[57]:


np.random.seed(0)


# In[58]:


(X_train, y_train), (X_test, y_test)= mnist.load_data()


# In[59]:


print(X_train.shape)
print(X_test.shape)


# In[60]:


assert(X_train.shape[0] == y_train.shape[0]), "The number of images is not equal to the number of labels."
assert(X_train.shape[1:] == (28,28)), "The dimensions of the images are not 28 x 28."
assert(X_test.shape[0] == y_test.shape[0]), "The number of images is not equal to the number of labels."
assert(X_test.shape[1:] == (28,28)), "The dimensions of the images are not 28 x 28."


# In[61]:


num_of_samples=[]
 
cols = 5
num_classes = 10

fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5,10))
fig.tight_layout()

for i in range(cols):
    for j in range(num_classes):
      x_selected = X_train[y_train == j]
      axs[j][i].imshow(x_selected[random.randint(0,(len(x_selected) - 1)), :, :], cmap=plt.get_cmap('gray'))
      axs[j][i].axis("off")
      if i == 2:
        axs[j][i].set_title(str(j))
        num_of_samples.append(len(x_selected))


# In[62]:


print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of the train dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()


# In[63]:


X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)


# In[64]:


y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


# In[65]:


X_train = X_train/255
X_test = X_test/255


# In[66]:


from tensorflow.keras.layers import Flatten


# In[67]:


from tensorflow.keras.layers import Conv2D


# In[68]:


from tensorflow.keras.layers import MaxPooling2D


# In[69]:


def leNet_model():
    model= Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# In[70]:


model=leNet_model()
print (model.summary())


# In[71]:


h=model.fit(X_train, y_train, epochs=10, validation_split=0.1, batch_size=400, verbose=1, shuffle=1)


# In[72]:


plt.plot(h.history['acc'])
plt.plot(h.history['val_acc'])
plt.legend(['acc', 'val_acc'])
plt.title('accuracy')
plt.xlabel('epoch')


# In[73]:


plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.title('loss')
plt.xlabel('epoch')


# In[81]:


import requests
from PIL import Image
import cv2


# In[88]:


url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcST8KzXHtkSHcxzdpnllMhAj0upLEwnNFdtY6j4YUPcmaf4Ty3u'
response = requests.get(url, stream=True)
#print(response)


# In[89]:


img = Image.open(response.raw)
plt.imshow(img)


# In[90]:


img_array = np.asarray(img)
resized =cv2.resize(img_array, (28, 28))
#print(resized.shape)
gray_scale = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
image =cv2.bitwise_not(gray_scale)
plt.imshow(image, cmap=plt.get_cmap("gray"))


# In[91]:


image = image/255
image =image.reshape(1, 28, 28, 1)


# In[92]:


prediction=model.predict_classes(image)
print ("predicted digit:", str(prediction))


# In[93]:


score =model.evaluate(X_test, y_test, verbose=0)
print('test score', score[0])
print('test accuracy', score[1])

