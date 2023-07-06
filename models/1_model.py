from pathlib import Path
import sys

CURRENT_DIR = Path('.').resolve()
MODULES_DIR = CURRENT_DIR.joinpath('src')
sys.path.append(str(MODULES_DIR))

import utils

from keras.datasets import imdb
import matplotlib.pyplot as plt
import seaborn as sns


def model():
    X_train, y_train, X_test, y_test = utils.import_imdb_dataset()
    id_to_word = utils.dic_words()
        
if __name__ == '__main__':
    model()