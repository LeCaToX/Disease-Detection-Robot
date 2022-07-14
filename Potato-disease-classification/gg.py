from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import requests

# Init
PLANT_LIST = ["Tomato","Pepper"]

model = {}

# Load model
for i in range(len(PLANT_LIST)):
    plant_name = PLANT_LIST[i]
    
    path = plant_name + '/keras_model.h5'
    model[i] = load_model(path)

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

while True:
    plant_id = int(input("Enter plant's id (0. Tomato, 1. Pepper): "))
    url = input("Enter image's URL: ")

    # Replace this with the path to your image
    image = Image.open(requests.get(url, stream=True).raw)
    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model[plant_id].predict(data)
    print(prediction)
