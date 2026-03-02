#aim:Build a Convolutional Neural Network for MNIST Handwritten Digit Classification
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
#Step-1 Data Preparation - Pixel values
#Load the data and split into training and testing data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#Reshape the image to 4-D format, including the channel dimension
x_dt = x_test.reshape ((x_test.shape[0]),28,28,1)
#Normalize the pixel values to the range [0,1]
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
#Perform one-hot encoding to convert labels to categorical (0/1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
#Step-2 Build the CNN model

model = models.Sequential()
# First convolutional layer
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
# Second convolutional layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
# Third convolutional layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# Flatten the output to feed it into a dense layer
model.add(layers.Flatten())
# Fully connected layer
model.add(layers.Dense(64, activation='relu'))
# Output layer with softmax activation for classification
model.add(layers.Dense(10, activation='softmax'))
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#Step-3 Training and Evaluation
#Train the model
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('Model Accuracy.tiff')
plt.show()
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('Model loss.tiff')
plt.show()


























