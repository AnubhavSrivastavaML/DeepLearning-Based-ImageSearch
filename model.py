from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from keras.utils.vis_utils import plot_model
from tensorflow.keras.utils import plot_model



def feature_extractor():
	model = VGG16(weights = 'imagenet')
	model = Model(inputs = model.input, outputs = model.layers[-2].output)
	plot_model(model, to_file="model.png", show_shapes=True)
	print("           MODEL ARCHITECTURE        ")
	print(model.summary())
	return model
	

	

