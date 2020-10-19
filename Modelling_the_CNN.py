#!/usr/bin/env python
# coding: utf-8

# In[7]:


from keras.models import Sequential
from keras.layers import Convolution2D,MaxPooling2D,Dense,Activation,Flatten,Dropout,Input
from keras.callbacks import ModelCheckpoint
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[8]:


X=np.load('data.npy')
Y=np.load('target.npy')


# In[9]:


print(X.shape,Y.shape)


# In[10]:


model=Sequential()
model.add(Input(shape=(100,100,1)))
model.add(Convolution2D(32,(3,3),activation='relu'))
model.add(Convolution2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))

model.add(Convolution2D(32,(5,5),activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(2,activation='softmax'))

model.summary()


# In[11]:


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[12]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1)
print("Training Data->",X_train.shape,Y_train.shape)
print("Testing Data->",X_test.shape,Y_test.shape)


# In[16]:


checkpoint = ModelCheckpoint('model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
history=model.fit(X_train,Y_train,epochs=10,callbacks=[checkpoint],validation_split=0.2,shuffle=True)


# In[ ]:





# In[30]:


plt.style.use("seaborn")
plt.plot(history.history['loss'],'r',label='training loss')
plt.plot(history.history['val_loss'],label='validation loss')
plt.xlabel('# epochs')
plt.ylabel('LOSS')
plt.legend()
plt.show()


# In[31]:


plt.plot(history.history['accuracy'],'r',label='training accuracy')
plt.plot(history.history['val_accuracy'],label='validation accuracy')
plt.xlabel('# epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


# In[32]:


results = model.evaluate(X_test,Y_test)


# In[33]:


print("Final Loss and Accuracy Respectively on test data->",results)


# In[34]:


model.save('./final_model')


# In[37]:


temp = model.predict(X_train[0].reshape((1,100,100,1)))


# In[40]:


ans = np.argmax(temp,axis=1)
print(ans)


# In[41]:


Y_train[0]


# In[12]:





# In[ ]:




