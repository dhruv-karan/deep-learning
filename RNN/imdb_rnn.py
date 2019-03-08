# -*- coding: utf-8 -*-

from keras.datasets import imdb
from keras.preprocessing import sequence

max_features=10000
max_len= 500
batch_size=32

(input_train,y_train),(input_test,y_test) = imdb.load_data(num_words=max_features)


print(len(input_train))
print(len(input_test))

len(input_train[0])

input_train = sequence.pad_sequences(input_train,maxlen=max_len)
input_test = sequence.pad_sequences(input_test,maxlen=max_len)
len(input_train[0])

from keras.layers import Dense
from keras.layers import Embedding, SimpleRNN
from keras.models import Sequential

model = Sequential()

model.add(Embedding(max_features,32))
model.add(SimpleRNN(32))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history = model.fit(input_train,y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)

#plotin gresults


import matplotlib.pyplot as plt

acc =  history.history['acc']
val_acc= history.history['val_acc']
loss= history.history['loss']
val_loss= history.history['val_loss']

epochs= range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label = 'Trainig_acc')
plt.plot(epochs,val_acc,'b',label = 'Validation acc')
plt.title('Train VS Validation acc')
plt.legend()
plt.figure()


plt.plot(epochs,loss,'bo',label = 'Training loss')









