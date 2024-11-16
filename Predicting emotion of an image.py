from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model("my_model.h5")

import cv2

# Load the image in grayscale
img_path="sad.jpg"
img = image.load_img(img_path, color_mode="grayscale", target_size=(48, 48))

# Convert the image to a NumPy array
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)  # Add a batch dimension

# Perform prediction
predictions = model.predict(img)


# Make predictions
emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
predictions = model.predict(img)
predicted_class = np.argmax(predictions)
predicted_emotion = emotions[predicted_class]

print(f'Predicted Emotion: {predicted_emotion}')
