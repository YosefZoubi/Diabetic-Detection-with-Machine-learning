from tensorflow import keras
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import MaxPooling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
import tensorflow.keras.utils as utils
from sklearn.metrics import recall_score, precision_score, accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import datetime


'''
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/eye_dr_cnn/src/Graph/
'''

postfix_file = str(datetime.datetime.now().timestamp())
local_path = '/content/drive/MyDrive/eye_dr_cnn/'


def load_processed_data(label_file, npy_file):
	# Import data
	labels = pd.read_csv(local_path + 'labels/' + label_file)
	X = np.load(local_path + 'data/' + npy_file)
	return X, labels


def get_train_test_data(label_file, npy_file, img_rows, img_cols, channels, test_data_size):
	"""
	Reshapes the data into format for CNN.

	INPUT
		label_file: labels file of npy_file
		npy_file: NumPy array of arrays, content of label_file
		img_rows: Image height
		img_cols: Image width
		channels: Specify if the image is grayscale (1) or RGB (3)
		test_data_size: size of test/train split. Value from 0 to 1

	OUTPUT
		(X_train, y_train), (X_test, y_test), input_shape, nb_classes
		X_train: Array of NumPy arrays
		X_test: Array of NumPy arrays
		y_train: Array of labels
		y_test: Array of labels
		input_shape: (height, weight, channels), weight: image heightXweight, channels: Specify if the image is grayscale (1) or RGB (3)
		nb_classes: Number of classes for classification
	"""

	X, labels = load_processed_data(label_file, npy_file)
	y = np.array(labels['level'])

	print("Splitting data into test/ train datasets")
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_data_size, random_state = 42)

	print("Reshaping Data")
	X_train = reshape_data(X_train, img_rows, img_cols, channels)
	X_test = reshape_data(X_test, img_rows, img_cols, channels)

	print("X_train Shape: ", X_train.shape)
	print("X_test Shape: ", X_test.shape)

	input_shape = (img_rows, img_cols, channels)

	print("Normalizing Data")
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')

	X_train /= 255
	X_test /= 255

	y_train = utils.to_categorical(y_train)
	y_test = utils.to_categorical(y_test)
	print("y_train Shape: ", y_train.shape)
	print("y_test Shape: ", y_test.shape)
	
	nb_classes = y_test.shape[1]
	print("Number of classes: ", nb_classes)
	
	assert nb_classes == len(set(y))

	input_shape = (img_rows, img_cols, channels)
	weights = class_weight.compute_class_weight('balanced', np.unique(y), y)

	return (X_train, y_train), (X_test, y_test), input_shape, nb_classes, weights


def reshape_data(arr, img_rows, img_cols, channels):
	"""
	Reshapes the data into format for CNN.

	INPUT
		arr: Array of NumPy arrays.
		img_rows: Image height
		img_cols: Image width
		channels: Specify if the image is grayscale (1) or RGB (3)

	OUTPUT
		Reshaped array of NumPy arrays.
	"""
	return arr.reshape(arr.shape[0], img_rows, img_cols, channels)


def create_dr_model_based_vgg16(input_shape):
	#using pre-trained model VGG16
	base_model = keras.applications.VGG16(input_shape=input_shape, weights='imagenet', include_top=False)
	base_model_w_top = keras.applications.VGG16(input_shape=input_shape, weights='imagenet', include_top=True)
	
	base_model_layers = list()
	for ilayer in range(1, len(base_model.layers)):
		base_model_layers.append(base_model.layers[ilayer])
	  
	base_model_top_layers = list()
	for ilayer in range(len(base_model.layers), len(base_model_w_top.layers) - 1): #without the last fc
		base_model_top_layers.append(base_model_w_top.layers[ilayer]) 

	new_model = list()
  
	for layer in base_model_layers:
		new_model.append(layer)
		new_model[-1].trainable=False

	batch_normalization_layer = keras.layers.BatchNormalization(
										name='batch_normalization_1',
										axis=-1,
										momentum=0.99,
										epsilon=0.001,
										center=True,
										scale=True,
										beta_initializer='zeros',
										gamma_initializer='ones',
										moving_mean_initializer='zeros',
										moving_variance_initializer='ones',
										beta_regularizer=None,
										gamma_regularizer=None,
										beta_constraint=None,
										gamma_constraint=None)
	
	new_model.append(batch_normalization_layer)

	for layer in base_model_top_layers:
		new_model.append(layer)
		new_model[-1].trainable=True

	new_model_input_layer = keras.Input(shape=input_shape, name = 'input')
  
	new_model_outputs = new_model_input_layer
	for layer in new_model:
		new_model_outputs = layer(new_model_outputs)
  
	new_model_output_layer = Dense(nb_classes, activation='softmax', name='output')(new_model_outputs)

	model = Model(name="DR-Model-Based-VGG16", inputs=new_model_input_layer, outputs=new_model_output_layer)
	
	return model


def cnn_model(X_train, X_test, y_train, y_test, nb_epoch, batch_size, nb_classes, weights, input_shape):
	'''
	Define and run the Convolutional Neural Network

	INPUT
		X_train: Array of NumPy arrays
		X_test: Array of NumPy arrays
		y_train: Array of labels
		y_test: Array of labels
		nb_epoch: Number of epochs
		batch_size: Batch size for the model
		nb_classes: Number of classes for classification
		input_shape: (height, weight, channels), weight: image heightXweight, channels: Specify if the image is grayscale (1) or RGB (3)

	OUTPUT
		Fitted CNN model
	'''

	model = create_dr_model_based_vgg16(input_shape)
	model.summary()

	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	
	stop = EarlyStopping(monitor='val_accuracy',
						min_delta=0.0001,
						patience=20,
						verbose=0,
						mode='auto')

	tensor_board = TensorBoard(log_dir=local_path + 'src/Graph', histogram_freq=0, write_graph=True, write_images=True)
	weight_temp = {i : weights[i] for i in range(5)}

	model_training_history = model.fit(X_train,y_train, batch_size=batch_size,
				epochs=nb_epoch,
				verbose=1,
				validation_split=0.2,
				class_weight=weight_temp,
				callbacks=[stop, tensor_board])

	return model, model_training_history


def show_plots(model_training_history, confusionMatrix, y_test, y_score):
	tm_file = local_path + 'src/plots/' + 'training_history_' + postfix_file + '.png'
	plt.figure()
	plt.plot(model_training_history.history['accuracy'])
	plt.plot(model_training_history.history['val_accuracy'])
	plt.title('Model accuracy of DR-Model-Based-VGG16')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.savefig(tm_file)
	plt.show()

	tm_loss_file = local_path + 'src/plots/' + 'loss_history_' + postfix_file + '.png'
	plt.figure()
	plt.plot(model_training_history.history['loss'])
	plt.plot(model_training_history.history['val_loss'])
	plt.title('Model loss of DR-Model-Based-VGG16')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.savefig(tm_loss_file)
	plt.show()

	cm_file = local_path + 'src/plots/' + 'confusionmatrix_' + postfix_file + '.png'
	plot_confusion_matrix(conf_mat=confusionMatrix, show_absolute=True, show_normed=True)
	plt.savefig(cm_file)
	plt.show()

	auc_file = local_path + 'src/plots/' + 'auc_' + postfix_file + '.png'
	fpr, tpr, thresholds = roc_curve(y_test, y_score, pos_label=2)
	plt.plot(fpr, tpr, color='blue',
					label='Model Prediction (AUC = {0:0.2f})'
					''.format(auc(fpr, tpr)))
	plt.plot([0, 1], [0, 1])
	plt.xlim([-0.05, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Model Prediction with AUC scores')
	plt.legend(loc="upper left")
	plt.savefig(auc_file)
	plt.show()

	print("trining plot: " + tm_file)
	print("trining loss plot: " + tm_loss_file)
	print("confusion matrix plot: " + cm_file)
	print("auc plot: " + auc_file)


if __name__ == '__main__':
	label_file = 'trainLabels_new_master_224_v2.csv'
	npy_file = 'X_train_224.npy'
	#dict_labels = {0:"No DR", 1:"Mild", 2:"Moderate", 3:"Severe", 4:"Proliferative DR"}
	img_rows, img_cols, channels = 224, 224, 3
	batch_size = 256
	nb_epoch = 25
	test_data_size = 0.2

	(X_train, y_train), (X_test, y_test), input_shape, nb_classes, weights = get_train_test_data(label_file, npy_file, img_rows, img_cols, channels, test_data_size)

	print("Training Model")
	model, model_training_history = cnn_model(X_train, X_test, y_train, y_test, nb_epoch, batch_size, nb_classes, weights, input_shape) 

	print("Predicting")
	y_pred = model.predict(X_test)

	score = model.evaluate(X_test, y_test, verbose=0)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])

	y_test = np.argmax(y_test, axis=1)
	y_pred = np.argmax(y_pred, axis=1)

	confusionMatrix = confusion_matrix(y_test, y_pred)
	print("Confusion matrix: \n" + str(confusionMatrix))
	print(classification_report(y_test, y_pred, digits=4))

	show_plots(model_training_history, confusionMatrix, y_test, y_pred)
	model_file = local_path + 'models/' + "DR_5_Classes_" + postfix_file + ".h5"
	model.save(model_file)
	print("Model saved: " + model_file)
	print("Completed")

