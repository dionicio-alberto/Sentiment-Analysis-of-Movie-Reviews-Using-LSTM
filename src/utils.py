from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def import_imdb_dataset():
    training_set, testing_set = imdb.load_data(num_words=10000)
    X_train, y_train = training_set
    X_test, y_test = testing_set
    return X_train, y_train, X_test, y_test

def dic_words():
    word_to_id = imdb.get_word_index()
    word_to_id = {
        key: (value+3) for key, value in word_to_id.items()
    }
    word_to_id['<PAD>'] = 0
    word_to_id['<START>'] = 1
    id_to_word = {
        value: key for key,value in word_to_id.items()
    }
    return id_to_word

def train_model(Optimizer, X_train, y_train, X_val, y_val):
    model = Sequential()
    model.add(Embedding(input_dim = 10000, output_dim = 128))
    model.add(LSTM(units=128))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(
        loss='binary_crossentropy', optimizer = Optimizer, 
        metrics=['accuracy'])
    scores = model.fit(
        X_train, y_train, batch_size=128, 
        epochs=10, validation_data=(X_val, y_val),
        shuffle=True)
    return scores, model

def plot_accuracy(score):
    # Plot accuracy per epoch
    plt.plot(range(1,11), score.history['accuracy'],
    label='Training Accuracy')
    plt.plot(range(1,11), score.history['val_accuracy'],
    label='Validation Accuracy')
    plt.axis([1, 10, 0, 1])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train and Validation Accuracy using RMSprop Optimizer')
    plt.legend()
    plt.show()
    

def plot_cm(y_test_pred,y_test):
    y_classes = (y_test_pred > 0.5).astype("int32")
    c_matrix = confusion_matrix(y_test,y_classes)
    ax = sns.heatmap(
        c_matrix, annot=True, 
        xticklabels=['Negative Sentiment','Positive Sentiment'],
        yticklabels=['Negative Sentiment', 'Positive Sentiment'],
        cbar=False, cmap='Blues', fmt='g')
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Actual")
    plt.show()
    
