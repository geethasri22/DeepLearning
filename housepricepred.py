#Import the reuried libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
#Load the dataset and normalize the values
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
mean = x_train.mean(axis=0); std = x_train.std(axis=0)
#Model the Network
model = keras.Sequential([
layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
layers.Dense(64, activation='relu'),
layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
#Train and evaluate the model
history = model.fit(x_train, y_train, epochs=150, batch_size=32, validation_split=0.1)
test_loss, test_mae=model.evaluate(x_test, y_test)
print(f"Test_MSE:{test_loss:0.4f}")
print(f"Test_MAE:{test_mae:0.4f}")
#Plot the graphs
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model Loss - MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('Model Loss.tiff')
plt.show()
