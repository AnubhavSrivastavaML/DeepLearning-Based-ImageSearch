import cv2
import numpy as np
import glob
from tqdm import tqdm



def create_encoddings_database(path,model):
    files = glob.glob(path)
    print("{} files found".format(len(files)))
    x = []
    images=[]
    bar = tqdm(total = len(files),position=0)
    for image in files:
        images.append(image)
        image = cv2.resize(cv2.imread(image),(224,224))
        x.append(image)
        bar.update()
    X = np.array(x)
    print("Shape for dataset")
    encoddings = model.predict(X)
    return encoddings,images

