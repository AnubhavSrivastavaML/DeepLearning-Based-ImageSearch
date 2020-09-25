import sqlite3
from get_batch_features import create_encoddings_database
from model import feature_extractor
from tqdm import tqdm
import json
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--imagefolder',default='database/*')
args = parser.parse_args()
#print(args.imagefolder)


create_table_query = "CREATE TABLE IF NOT EXISTS tblRequest(id integer PRIMARY KEY , path text, encoddings blob)"
sqlite_insert_with_param = """INSERT INTO 'tblRequest' ('id', 'path', 'encoddings') VALUES (?, ?, ?);"""

print("Loading feature extractor ... ")
feature = feature_extractor()
print("Feature Entractor loaded")
 

path = args.imagefolder+'/*'
encoddings , images = create_encoddings_database(path,feature)

	


print("Inserting {} into Database".format(len(images)))
pbar = tqdm(total = len(images),position=0)
for i in range (len(images)):    
    with sqlite3.connect("imagedata") as conn:
            conn.execute(sqlite_insert_with_param , (i,images[i],json.dumps(encoddings[i].tolist())))
            		
            pbar.update()
            

	


conn = sqlite3.connect("imagedata")
c = conn.cursor()
c.execute(create_table_query)
c.close()

