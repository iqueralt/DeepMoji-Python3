# -*- coding: utf-8 -*-

""" Use DeepMoji to score texts for emoji distribution.

The resulting emoji ids (0-63) correspond to the mapping
in emoji_overview.png file at the root of the DeepMoji repo.

Writes the result to a csv file.
"""
from __future__ import print_function, division
import example_helper
import json
import csv
import numpy as np
import tensorflow as tf
from flask import Flask
from flask_cors import cross_origin

from deepmoji.sentence_tokenizer import SentenceTokenizer
from deepmoji.model_def import deepmoji_emojis
from deepmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH, EMOJI_ENCODING

app = Flask(__name__)

def top_elements(array, k):
    """Auxiliar function that returns top elements"""
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]

# Global variables needed for the models
maxlen = 30
batch_size = 32

print('Loading model from {}.'.format(PRETRAINED_PATH))
model = deepmoji_emojis(maxlen, PRETRAINED_PATH)
model.summary()
graph = tf.get_default_graph()

with open(VOCAB_PATH, 'r') as f:
    vocabulary = json.load(f)
st = SentenceTokenizer(vocabulary, maxlen)

@app.route("/analyze/<comment>")
@cross_origin()
def show_post(comment):
    """Returns sentiment analysis for a particular string"""
    global graph
    with graph.as_default():
        tokenized, _, _ = st.tokenize_sentences([comment])
        print('Running predictions.')
        prob = model.predict(tokenized)

        # Find top emojis for each sentence. Emoji ids (0-63)
        # correspond to the mapping in emoji_overview.png
        # at the root of the DeepMoji repo.

        print("here, here")
        t_tokens = tokenized[0]
        t_prob = prob[0]
        ind_top = top_elements(t_prob, 10)
        emoji_list = [EMOJI_ENCODING[ind].decode('unicode_escape') for ind in ind_top]
        for element in emoji_list:
            print(element)

        return " ".join(emoji_list)
