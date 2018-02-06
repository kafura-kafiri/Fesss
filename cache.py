from utility.pqdict import pqdict
from keras.models import model_from_json
import os
import time
cache = pqdict(key=lambda model: model['lru'])
models_dir = 'models'
max_size = 2


def get(_id):
    """

    :param _id:
    :return: model
    """
    return cache.get(_id, _retrieve(cache, _id))['model']


def _retrieve(d, key):
    """
    retrieve from fs to ram.
    :param d:
    :param key:
    :return: model
    """
    # load json and create model
    json_path = os.path.join(models_dir, key, 'model.json')
    h5_path = os.path.join(models_dir, key, 'model.h5')
    with open(json_path, 'r') as j:
        loaded_model = model_from_json(j.read())
    # load weights into new model
    loaded_model.load_weights(h5_path)
    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    loaded_model = {
        'lru': time.clock(),
        'model': loaded_model
    }
    d[key] = loaded_model
    if len(d) > max_size:
        d.pop()
    return loaded_model
