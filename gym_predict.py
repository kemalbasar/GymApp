import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import argparse


parser = argparse.ArgumentParser()
a1 = parser.add_argument('--input', required=True, help='path to image folder')
args = parser.parse_args()

classifier = load_model('squat.h5')  # load the model that was created using cnn_multiclass.py

test_image = image.load_img(args.input,
                            target_size=(64, 64))  # folder predictions with images that I want to test
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = classifier.predict(test_image)  # returns array

if result[0][0] == 1:
    prediction = 'down'  # predictions in array are in alphabetical order
elif result[0][1] == 1:
    prediction = 'up'

print(result)
print("asdads")
print(prediction)
