import cv2
import argparse
import sqlite3
import json
import numpy as np
from get_features import feature
from model import feature_extractor
from sklearn.metrics.pairwise import cosine_similarity


parser = argparse.ArgumentParser()
parser.add_argument('--threshold',type = int ,default=0.5)
parser.add_argument('--image',required = True )

args = parser.parse_args()

extractor = feature_extractor()

encoddings = feature(args.image,extractor)

query = """ SELECT * FROM tblRequest"""
with sqlite3.connect("imagedata") as conn:
	data = conn.execute(query)
data = data.fetchall()
print("Fetched {} images from database".format(len(data)))
result=0
for d in data :
	en = np.expand_dims(np.array(json.loads(d[2])),axis=0)
	cosine = cosine_similarity(en,encoddings)
	if cosine > args.threshold :
		img = cv2.imread(d[1])
		cv2.imwrite('results/'+str(result)+'.jpg',img)
		result+=1
print("Found {} reults".format(result+1))
		



