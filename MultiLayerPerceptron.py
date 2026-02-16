#Aim: Implement Multi Layer Perceptron algorithm for MNIST Hand Written Digit Classification
#import librarires
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
#import dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data()
#normalise the pixel values
x_train=x_train.reshape(-1,28*28).astype("float32")/255
x_test=x_test.reshape(-1,28*28).astype("float32")/255
#model the network
model=keras.Sequential([
    layers.Dense(128,activation='relu',input_shape=(784,)),
    layers.Dense(64,activation='relu'),
    layers.Dense(10,activation='softmax')
])
#compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
#model evaluation
history=model.fit(x_train,y_train,epochs=5,batch_size=128,validation_split=0.1)
test_loss,test_acc=model.evaluate(x_test,y_test)
print(f"Test Accuracy:{test_acc:.4f}")
print(f"Test loss:{test_loss:.4f}")
#plot the graph
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epochs')
plt.savefig('Accuracy-MNIST.tiff')
plt.show()
