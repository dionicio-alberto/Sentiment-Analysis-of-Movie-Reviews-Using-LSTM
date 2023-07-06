from pathlib import Path
import sys
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dense

CURRENT_DIR = Path('.').resolve()
MODULES_DIR = CURRENT_DIR.joinpath('src')
sys.path.append(str(MODULES_DIR))

import utils
import random
from keras.datasets import imdb
from keras.utils import pad_sequences

def data():
    X_test, y_test,X_train, y_train = utils.import_imdb_dataset()


    X_test_padded = pad_sequences(X_test, maxlen = 100)
    X_train_padded = pad_sequences(X_train, maxlen = 100)
    return X_train_padded, y_train, X_test_padded, y_test
    
def results_error(y_test_predict):
    false_negatives = []
    false_positives = []
    for i in range(len(y_test_predict)):
        if y_test_predict[i][0] != y_test[i]:
            if y_test[i] == 0: # False Positive
                false_positives.append(i)
            else:
                false_negatives.append(i) 
                
    id_to_word = utils.dic_words() 
    print('False positive: ') 
    print(' '.join(id_to_word.get(i, 'UNK') for i in X_test_padded[random.choice(false_positives)]))
    print(' ')
    print('False negative: ')
    print(' '.join(id_to_word.get(i, 'UNK') for i in X_test_padded[random.choice(false_negatives)]))           
     
    
def rms(X_train_padded, y_train, X_test_padded, y_test):
    rmsprop_score, rmsprop_model = utils.train_model(
    Optimizer='RMSprop',
    X_train=X_train_padded,
    y_train=y_train,
    X_val= X_test_padded,
    y_val=y_test
    )    
    utils.plot_accuracy(rmsprop_score)
    y_test_predict_rms = rmsprop_model.predict(X_test_padded)
    utils.plot_cm(y_test_predict_rms,y_test)
    results_error(y_test_predict_rms)
    
def adam(X_train_padded, y_train, X_test_padded, y_test):
    adam_score, adam_model = utils.train_model(
    Optimizer='adam',
    X_train=X_train_padded,
    y_train=y_train,
    X_val= X_test_padded,
    y_val=y_test
    )
    utils.plot_accuracy(adam_score)
    y_test_predict_adam = adam_model.predict(X_test_padded)
    utils.plot_cm(y_test_predict_adam,y_test)
    results_error(y_test_predict_adam)
    
def sgd(X_train_padded, y_train, X_test_padded, y_test):
    sgd_score, sgd_model = utils.train_model(
    Optimizer='SGD',
    X_train=X_train_padded,
    y_train=y_train,
    X_val= X_test_padded,
    y_val=y_test
    )
    utils.plot_accuracy(sgd_score)
    y_test_predict_sgd = sgd_model.predict(X_test_padded)
    utils.plot_cm(y_test_predict_sgd,y_test)
    results_error(y_test_predict_sgd)
    
           
if __name__ == '__main__':
    X_train_padded, y_train, X_test_padded, y_test= data()
    sgd(X_train_padded, y_train, X_test_padded, y_test)
    rms(X_train_padded, y_train, X_test_padded, y_test)
    adam(X_train_padded, y_train, X_test_padded, y_test)
    