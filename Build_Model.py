import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer, text_to_word_sequence

# load data
data = pd.read_csv("Sentiment Analysis Dataset.csv", skiprows = [8835, 535881], usecols = [1, 3])
data

# Split data into training and testing data
x = list(data['SentimentText'])
y = list(data['Sentiment'])
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state = 0)


# create a new Tokenizer that finds the 3000 most popular words found in our dataset
tokenizer = Tokenizer(num_words = 3000)
tokenizer.fit_on_texts(train_x)

# save the dictionary
dictionary = tokenizer.word_index
with open('dictionary.json', 'w') as dictionary_file:
    json.dump(dictionary, dictionary_file)

train_WordIndices = []
# This converts strings of text into lists of index array
for text in train_x:
    wordIndices = [dictionary[word] for word in text_to_word_sequence(text)]
    train_WordIndices.append(wordIndices)

train_WordIndices_arr = np.asarray(train_WordIndices)

# create matrices out of the indexed tweets
# tokenizer.sequences_to_matrix returns a numpy matrix of (len(allWordindices), 3000)
train_x = tokenizer.sequences_to_matrix(train_WordIndices_arr, mode='binary')
train_y = keras.utils.to_categorical(train_y, 2)

# build model
model = Sequential()
model.add(Dense(512, input_shape=(3000,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')
model.fit(train_x, train_y, batch_size=32, epochs=5, verbose=1, validation_split=0.1, shuffle=True)

# Validation
val_WordIndices = []
def sentence_to_words(text) :
    wordIndices = []
    arr_word = text_to_word_sequence(text)
    for word in arr_word:
        if word in dictionary:
            wordIndices.append(dictionary[word])
    return wordIndices
            
# This converts strings of text into lists of index array
for text in val_x:
    val_WordIndices.append(sentence_to_words(text))

val_WordIndices_arr = np.asarray(val_WordIndices)
val_x = tokenizer.sequences_to_matrix(val_WordIndices_arr, mode='binary')
pred_y = model.predict(val_x)
val_y1 = keras.utils.to_categorical(val_y, 2)
# evaluation metrics
from sklearn.metrics import accuracy_score
accuracy_score(val_y1, pred_y.round())

#save the model
model.save('my_model.h5')

tokenizer2 = Tokenizer(num_words=3000)
labels = ['negative', 'positive']
while True:
    text_input = input('Evaluate this:')
    if len(text_input) == 0:
        break
    words_input = sentence_to_words(text_input)
    input1 = tokenizer2.sequences_to_matrix([words_input], mode='binary')
    pred = model.predict(input1)
    print("%s sentiment; %f%% confidence" % (labels[np.argmax(pred)], pred[0][np.argmax(pred)] * 100))

