from tensorflow import keras
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import sys
import datetime
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


postfix_file = str(datetime.datetime.now().timestamp())
local_path = '/content/drive/MyDrive/eye_dr_cnn/'


def load_image(img_path):
	img = image.load_img(img_path, target_size=(224, 224))
	img_tensor = image.img_to_array(img)                    # (height, width, channels)
	img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
	img_tensor = img_tensor.astype('float32')
	img_tensor /= 255.                                      # imshow expects values in the range [0, 1]
	return img_tensor


def predictImage(model, img):
	print(img)
	img = load_image(img)
	pred = model.predict(img)
	print("prediction: " + str(pred.argmax()))
	print("prediction result: " + str(pred[0][pred.argmax()]))
	return pred.argmax()


if __name__ == '__main__':
	model_file = 'DR_5_Classes_1608598532.691587.h5'

	print("Loading Model")
	model = load_model(local_path + 'models/' + model_file)

	print("Predicting")
	img = sys.argv[1] #local_path + 'data/sample/17_right.jpeg'
	predictImage(model, img)
	print("Completed")

