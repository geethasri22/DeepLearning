import tensorflow as tf
from tensorflow.keras.applications.vgg16 import (
    VGG16,
    preprocess_input,
    decode_predictions
)
from tensorflow.keras.preprocessing import image
import numpy as np
model = VGG16(weights='imagenet')
my_image_path = r"C:\Users\AIML\Desktop\car.jpg"  

def classify_image(img_path):
    try:
        # Load and resize image
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = model.predict(x)
        print(f"Predictions for: {img_path}")
        for pred in decode_predictions(preds, top=3)[0]:
            print(f"- {pred[1]}: {pred[2]*100:.2f}%")

    except Exception as e:
        print(f"Error: Could not find or load the image.\nDetails: {e}")

# 3. Run the function
classify_image(my_image_path)
