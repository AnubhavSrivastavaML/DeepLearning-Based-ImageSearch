import cv2
import numpy as np

def feature(image,model):
    image = cv2.resize(cv2.imread(image),(224,224))    
    image = np.expand_dims(image , axis=0)
    encoddings = model.predict(image)
    return encoddings
