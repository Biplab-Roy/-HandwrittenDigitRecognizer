import numpy as numpy
import tensorflow.keras.models
import tensorflow.keras
from tensorflow.keras.models import model_from_json
import tensorflow as tf
from tensorflow.python.framework import ops


def init():
    json_file = open(r'E:\Projects\NLP\Web\Digit Recognizer Website\model\model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(r'E:\Projects\NLP\Web\Digit Recognizer Website\model\model.h5')
    loaded_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    graph = ops.reset_default_graph()
    return loaded_model,graph