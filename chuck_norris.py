from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import urllib, json
import requests
from HTMLParser import HTMLParser

def getChuckNorrisJokesEN():
    url_count = "http://api.icndb.com/jokes/count"
    r = requests.get(url_count)
    data_count= r.json()
    sentences_count = int(data_count['value'])
    sentences_text = []
    for i in xrange(1,sentences_count):
        url='http://api.icndb.com/jokes/%d'%(i,)
        try:
            r=requests.get(url)
            data = r.json()
            if data['type']=='success':
                sentences_text.append(data['value']['joke'])
        except ValueError:
            continue
    text = '\n'.join(sentences_text)
    parser = HTMLParser()
    text = parser.unescape(text)
    text = text.encode("utf8").lower()
    return text

def getChuckNorrisJokesFR():
    iMax = 105
    lines = dict()
    for i in xrange(1, iMax):
        url = 'http://chucknorrisfacts.fr/api/get?data=type:text;nb:99;page:%d'%(i,)
        r = requests.get(url)
        data = r.json()
        for e in data:
            lines[e['id']]=e['fact']
    sentences_text= []        
    for j in lines.keys():
        sentences_text.append(lines[j])
    text = '\n'.join(sentences_text)
    parser = HTMLParser()
    text = parser.unescape(text)
    text = text.encode("utf8").lower()
    text = text.replace('<br />','')
    text = "\n".join([ll.rstrip() for ll in text.splitlines() if ll.strip()])
    return text

text = getChuckNorrisJokesFR()
print('corpus length:', len(text))
chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=128, nb_epoch=2)

    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

