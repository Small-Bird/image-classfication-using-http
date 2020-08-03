import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from werkzeug.utils import secure_filename
import json
import time

""" sqlite3 context manager and try-except error handling """

import sqlite3

database = 'predict_result.sqlite'

def create_table():
    try:
        with sqlite3.connect(database) as conn:
            conn.execute('create table if not exists info(time text, name text, accuracy real)')
    except sqlite3.Error as e:
        print(f'error creating table because {e}')
    finally:
        conn.close()


def add_data(time,name,accuracy):
    try:
        with sqlite3.connect(database) as conn:
            conn.execute("insert into info values('%s','%s','%f')"%(time,name,accuracy))
    except sqlite3.Error:
        print('Error adding rows')
    finally:
        conn.close()


def print_all_data():
    # Execute a query. Do not need a context manager, as no changes are being made to the db
    try:
        conn = sqlite3.connect(database) 
        for row in conn.execute('select * from info'):
            print(row)
    except sqlite3.Error as e:
        print(f'Error selecting data from info table because {e}')
    finally:
        conn.close()


def delete_table():
    try:
        with sqlite3.connect(database) as conn:
            conn.execute('drop table info')
    except sqlite3.Error as e:
        print(f'Error deleting info table because {e}')
    finally:
        conn.close()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

MODEL_PATH = 'models/your_model.h5'

model = ResNet50(weights='imagenet')

def model_predict(img_path, model):
  img = image.load_img(img_path, target_size=(224, 224))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x, mode='caffe')
  preds = model.predict(x)
  return preds


@app.route('/predict', methods=['GET', 'POST'])
def upload():
  if request.method == 'POST':
    f = request.files['image_file']
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
      basepath,
      'uploads',
      secure_filename(f.filename))
    f.save(file_path)
    preds = model_predict(file_path, model)
    pred_class = decode_predictions(preds)
    # database_prepare()

    firstName = str(pred_class[0][0][1])
    secondName = str(pred_class[0][1][1])
    thirdName = str(pred_class[0][2][1])
    
    firstAccuracy = float(pred_class[0][0][2])
    secondAccuracy = float(pred_class[0][1][2])
    thirdAccuracy = float(pred_class[0][2][2])

    add_data(time.time(),firstName,firstAccuracy)
    add_data(time.time(),secondName,secondAccuracy)
    add_data(time.time(),thirdName,thirdAccuracy)

    result = [
      [
        firstName,
        firstAccuracy        
      ],
      [
        secondName,
        secondAccuracy
      ],
      [
        thirdName,
        thirdAccuracy
      ],
    ]
    return jsonify(result)
  return None

if __name__ == '__main__':
  create_table()
  app.run(debug = False, threaded = False)
