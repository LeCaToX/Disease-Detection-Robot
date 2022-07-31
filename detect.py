from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import requests


# Init
PLANT_LIST = ["Tomato","Pepper","Potato"]

diseases_list = []


model = {}

# Load model
for i in range(len(PLANT_LIST)):
	plant_name = PLANT_LIST[i]
  
	path = plant_name + '/keras_model.h5'
	model[i] = load_model(path)

	# Load disease name
	path = plant_name + "/labels.txt"
    
	f = open(path, "r")
	disease = f.read().split('\n')

	diseases_list.append(disease)

print(diseases_list)

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

class classification:
	
	def detect(plant_id:int, img_url:str):
 	    # ID code: 0. Tomato, 1. Pepper, 2. Potato 

		plant_id = plant_id
		url = img_url

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
		prediction = diseases_list[plant_id][np.argmax(model[plant_id].predict(data))]

		return prediction

#print(classification.detect(1,"https://peppergeek.com/wp-content/uploads/2020/08/Pepper-Plant-Leaf-Spot.jpg"))
