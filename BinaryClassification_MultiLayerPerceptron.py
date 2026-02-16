#Aim:Implement MultiLayer Perceptron algorithm for Binary Calssification
#Import the required libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras import layers, models`
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
#Load the dataset and Perform Padding
num_words=10000
(x_train,y_train), (x_test,y_test)=imdb.load_data(num_words=num_words)
maxlen=200
x_train=pad_sequences(x_train, maxlen=maxlen)
x_test=pad_sequences(x_test, maxlen=maxlen)
#Model the network (bag of words style)
model=keras.Sequential([
layers.Embedding(num_words, 32, input_length=maxlen),
layers.GlobalAveragePooling1D(),
layers.Dense(16,activation='relu'),
layers.Dense(1,activation='sigmoid')
])

#Define the metrics
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
#Train and Evaluate the model
history=model.fit(x_train, y_train, epochs = 10, batch_size=120, validation_split=0.2)
test_loss, test_acc=model.evaluate(x_test,y_test)
print(f"Test accuracy:{test_acc:.4f}")
print(f"Test loss:{test_loss:.4f}")
#Plot the graph
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('Model Accuracy.tiff')
plt.show()
