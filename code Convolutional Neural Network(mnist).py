
# In[18]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import random


# In[19]:


np.random.seed(0)


# In[20]:

#lire mnist_data
(X_train, y_train), (X_test, y_test)= mnist.load_data()


# In[21]:


print(X_train.shape)
print(X_test.shape)


# In[22]:

#pour tester si toutes les données sont bien cohérentes
assert(X_train.shape[0] == y_train.shape[0]), "The number of images is not equal to the number of labels."
assert(X_train.shape[1:] == (28,28)), "The dimensions of the images are not 28 x 28."
assert(X_test.shape[0] == y_test.shape[0]), "The number of images is not equal to the number of labels."
assert(X_test.shape[1:] == (28,28)), "The dimensions of the images are not 28 x 28."


# In[23]:

#afficher 5 exemplaires pour chaque digit 
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


# In[24]:


print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of the train dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()


# In[25]:


X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)


# In[26]:


y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


# In[27]:


X_train = X_train/255
X_test = X_test/255


# In[28]:


from tensorflow.keras.layers import Flatten


# In[29]:


from tensorflow.keras.layers import Conv2D


# In[30]:


from tensorflow.keras.layers import MaxPooling2D


# In[31]:

# on va créer un réseau qui contient deux couches convolutions / 2 couches polling / une couche de 500 noeuds (fully connected layer) sortie 10 noeuds pour traiter les images de digits 
#pour plus des détailles voir l'image "Network parameters summary"
def leNet_model():
    model= Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu')) 
	# on applique 30 filtres de taille (5, 5) sur les images de taille (nombre des lignes=28, nombres des colones= 28, profondeur=1) aprés la convolution on l'applique à une fonction non linéare relu
	model.add(MaxPooling2D(pool_size=(2, 2)))
	#max polling : permet de réduire les nombre des pixels à traiter toute en gardant les détailles de filtrage à condition que les tailles soit raisonnable 
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
	# il faut aplatie les doonées afin de l'appliquer au couches connectés (connected layers)
    model.add(Dense(500, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# In[32]:


model=leNet_model()
print (model.summary())


# In[34]:

# entrainnement de réseau 

h=model.fit(X_train, y_train, epochs=10, validation_split=0.1, batch_size=400, verbose=1, shuffle=1)


# In[36]:

# afficher les précisions de résaeu à l'égard de données de  validation et d'entrainnement 
plt.plot(h.history['acc'])
plt.plot(h.history['val_acc'])
plt.legend(['acc', 'val_acc'])
plt.title('accuracy')
plt.xlabel('epoch')


# In[37]:

# afficher les erreurs des validation et d'entrainnement 
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.title('loss')
plt.xlabel('epoch')

# prédiction des test_data

score =model.evaluate(X_test, y_test, verbose=0)
print('test score', score[0])
print('test accuracy', score[1])

 
# In[38]:

#pour tester la prédiction de réseau trainé on prend une URL d'une image qui contient un digit et on fait tous les traitements possibles (conversion en gray, redimensionnement) afin de l'appliquer à notre réseau
import requests
from PIL import Image
import cv2


# In[49]:


url =  'https://www.researchgate.net/profile/Jose_Sempere/publication/221258631/figure/fig1/AS:305526891139075@1449854695342/Handwritten-digit-2.png'
response = requests.get(url, stream=True)
#print(response)


# In[50]:


img = Image.open(response.raw)
plt.imshow(img)


# In[51]:


img_array = np.asarray(img)
resized =cv2.resize(img_array, (28, 28))
#print(resized.shape)
gray_scale = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
image =cv2.bitwise_not(gray_scale)
plt.imshow(image, cmap=plt.get_cmap("gray"))


# In[52]:


image = image/255
image =image.reshape(1, 28, 28, 1)


# In[53]:


prediction=model.predict_classes(image)
print ("predicted digit:", str(prediction))

