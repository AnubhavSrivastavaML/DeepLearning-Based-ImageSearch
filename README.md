# DeepLearning-Based-ImageSearch

The repository provides complete implmentation of searching images through deep learning techniques.The implementation can be used for searching similar images through large image databases

### Requirements: 

1.Tensorflow

2.Keras

3.OpenCV

4.Glob

5.Json

6.Sqlite3

7.Numpy

8.tqdm

Use this repository to create the image search application. It includes sqlite3 database creation with named 'imagedata' with table ('id','path of image',features)

## Usage

To create database place images in database folder and run : python3 createdatabase.py

To search for similar images from image run : python3 search_image_database.py --image <path/to/your/image>
