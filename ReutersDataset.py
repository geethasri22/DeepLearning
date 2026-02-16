#aim:Design a Neural Network for CLassifying news wires(multi class classification) using Reuters data set
#import the required modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import reuters
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
#import the dataset
num_words = 10000
(x_train, y_train), (x_test, y_test)=reuters.load_data(num_words=num_words)
#multi-hot-encoding of words
def vectorize(seqs, dim=num_words):
    res=np.zeros((len(seqs), dim), dtype="float32")
    for i, seq in enumerate(seqs):
        res[i, seq] = 1.0
    return res
x_train = vectorize(x_train)
x_test = vectorize(x_test)
#one-hot-encoding of words
y_train_oh = to_categorical(y_train)
y_test_oh = to_categorical(y_test)
#model the network
model = keras.Sequential([
layers.Dense(64, activation='relu', input_shape=(num_words,)),
layers.Dense(64, activation='relu'),
layers.Dense(46, activation='softmax')
])
model.compile(optimizer='adam',
loss='categorical_crossentropy',metrics=['accuracy'])
#train and evaluate the model

history=model.fit(x_train, y_train_oh, epochs=10, batch_size=512,
validation_split=0.15)
test_loss, test_acc=model.evaluate(x_test, y_test_oh)
print(f"Test accuracy:{test_acc}")
print(f"Test loss:{test_loss}")
#plot the graph
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('Model Accuracy.jpg')
plt.show()
