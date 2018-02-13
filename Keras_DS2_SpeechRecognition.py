#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import os
import datetime
import pickle
import shutil
import subprocess
import time
import numpy as np
import csv
import pandas as pd

import keras
import matplotlib.pyplot as plt
from keras.callbacks import *
from keras.layers import *
from keras.models import *
# from keras.utils.visualize_util import plot
from keras.utils import plot_model  # pip install pydot-ng
from keras.backend.tensorflow_backend import set_session

import numpy as np
import python_speech_features
import scipy.io.wavfile as wav

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import tensor_util

from util.audio import audiofile_to_input_vector
from util.text import Alphabet
from util.text import sparse_tensor_value_to_texts, wer, Alphabet, ndarray_to_text

import matplotlib.pyplot as plt
from matplotlib import cm

global doTraining
doTraining = True

global use_ds2
use_ds2 = True  # Deep Speech 2

# Number of MFCC features
global n_input
n_input = 40  # MFCC, maybe need add Delta

# The number of frames in the context
global n_context
n_context = 0

global feature_len
feature_len = 100  # all input is 1s wavfiel 1000ms/10ms = 100.

global feature_dim
feature_dim = n_input * (n_context * 2 + 1)

global alphabet
alphabet = Alphabet('./alphabet.txt')
print('alphabet.size() ', alphabet.size())
print(alphabet._label_to_str)
# The number of characters in the target language plus one
global n_character
n_character = alphabet.size() + 1  # +1 for CTC blank label

global max_labellen
max_labellen = 6

global n_hidden
n_hidden = 128

trfile = 'data/speechcmd_train.csv'
cvfile = 'data/speechcmd_dev.csv'
testfile = 'data/speechcmd_test.csv'

NB_EPOCH = 15
BATCH_SIZE = 128
BATCH_SIZE_INFERENCE = 1024
VERBOSE = 1


def loadfromecsv(fname):
    r = []
    with open(fname) as docfile:
        reader = csv.reader(docfile)
        for line in reader:
            r.append(line)
    return r


def _get_files_mfcc(wav_filenames):
    # print('Processing MFCC...')
    mfccs = []
    lens = []
    for audio_fname in wav_filenames:
        this_mfcc = audiofile_to_input_vector(audio_fname, n_input, n_context)
        if len(this_mfcc) != feature_len:
            needlen = feature_len - len(this_mfcc)
            a = ([[0 for x in range(feature_dim)] for y in range(needlen)])
            this_mfcc = np.concatenate((this_mfcc, np.array(a)))
        # print(this_mfcc.shape)
        this_mfcc = np.reshape(this_mfcc, (feature_len, n_input, 1))
        mfccs.append(this_mfcc)
        lens.append(len(this_mfcc))
    a_mfccs = np.array(mfccs)  # shape, (batch, time_step_len, feature_len)
    a_lens = np.array(lens)  # shape, (batch, 1), value == time_step_len
    # print('MFCCs shape', a_mfccs.shape, a_lens.shape)
    return a_mfccs, a_lens


class dataGen_mfcc_ctc():
    def __init__(self, csvfile, batchSize=128):
        self.batchPointer = 0
        self.data = loadfromecsv(csvfile)[1:]
        self.batchSize = batchSize
        self.totallen = int(len(self.data))
        if doTraining:
            self.numSteps = int(self.totallen / self.batchSize + 0)
        else:
            self.numSteps = int(self.totallen / self.batchSize + 1)
        print('dataGen_speechcmd_mfcc: init', len(self.data))
    
    def __next__(self):
        if (self.batchPointer + self.batchSize) >= len(self.data):
            if not doTraining:
                thislen = len(self.data) - self.batchPointer
            else:  # enable this for fixed size, when training.
                self.batchPointer = 0
                thislen = self.batchSize
        else:
            thislen = self.batchSize
        a_mfccs, a_mfcclens, a_label, a_labelen, wav_filenames = self.getNextSplitData(
            self.data[self.batchPointer: self.batchPointer + thislen])
        self.batchPointer += thislen
        if self.batchPointer >= len(self.data):
            self.batchPointer = 0
            np.random.shuffle(self.data)
        return a_mfccs, a_mfcclens, a_label, a_labelen, wav_filenames
    
    def _getLabel(self, transcripts):
        a_label = np.asarray(
            [[alphabet.label_from_string(a) for a in c + ' ' * (max_labellen - len(c))] for c in transcripts])
        a_labelen = np.asarray([len(c) for c in transcripts])
        return a_label, a_labelen
    
    def getNextSplitData(self, fileinfos):
        wav_filenames = list(zip(*fileinfos))[0]
        # wav_filesizes = list(zip(*fileinfos))[1]
        transcripts = list(zip(*fileinfos))[2]
        a_mfccs, a_mfcclens = _get_files_mfcc(wav_filenames)
        a_label, a_labelen = self._getLabel(transcripts)
        return a_mfccs, a_mfcclens, a_label, a_labelen, wav_filenames


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def gen_ctc_byclass(dataGen, return_wavfilenames=False):
    batch_size = dataGen.batchSize
    while True:
        a_mfccs, a_mfcclens, a_label, a_labelen, wavfilenames = dataGen.__next__()
        if not return_wavfilenames:
            yield [a_mfccs, a_label, a_mfcclens, a_labelen], np.ones(batch_size)
        else:
            yield [a_mfccs, a_label, a_mfcclens, a_labelen], np.ones(batch_size), wavfilenames


def evaluate_CTC(base_model, batch_num=10):
    batch_acc = 0
    testGen = dataGen_mfcc_ctc(testfile, batchSize=BATCH_SIZE)
    for i in range(batch_num):
        y_pred = base_model.predict(X_test)
        shape = y_pred[:, 2:, :].shape
        ctc_decode = K.ctc_decode(y_pred[:, 2:, :],
                                  input_length=np.ones(shape[0]) * shape[1])[0][0]
        out = K.get_value(ctc_decode)[:, :4]
        if out.shape[1] == 4:
            batch_acc += ((y_test == out).sum(axis=1) == 4).mean()
    return batch_acc / batch_num


class Evaluate(Callback):
    def __init__(self, base_model):
        self.accs = []
        self.base_model = base_model
    
    def on_epoch_end(self, epoch, logs=None):
        print('Evaluate...')
        acc = evaluate_CTC(self.base_model) * 100
        self.accs.append(acc)
        print
        print('Evaluate acc: %f%%' % acc)


# ref https://github.com/mozilla/DeepSpeech/pull/938/commits/72a6331e808efbdf93715f4ac5ea9226ad202818
# Epoch 15/20
# - 269s - loss: 0.1738 - val_loss: 0.8504
# param: 1367k
# acc. ~94%
def do_train_ds2_CTC(cont_from_checkpointfile=None):
    rnn_size = n_hidden
    
    trainGen = dataGen_mfcc_ctc(trfile, batchSize=BATCH_SIZE)
    validGen = dataGen_mfcc_ctc(cvfile, batchSize=BATCH_SIZE)
    
    input_tensor = Input(name='the_input', shape=[feature_len, feature_dim, 1])
    x = input_tensor
    x_shape = x.get_shape()
    print(x.get_shape())
    x = Convolution2D(filters=32, kernel_size=(11, 3), strides=1, activation='relu', padding='same')(x)
    # print(x.get_shape())
    conv_shape = x.get_shape()
    x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(x)
    x = Dropout(0.1)(x)
    # print(x.get_shape())
    
    gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(x)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True,
                 kernel_initializer='he_normal', name='gru1_b')(x)
    x = concatenate([gru_1, gru_1b])
    
    gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(x)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True,
                 kernel_initializer='he_normal', name='gru2_b')(x)
    x = concatenate([gru_2, gru_2b])
    
    gru_3 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru3')(x)
    gru_3b = GRU(rnn_size, return_sequences=True, go_backwards=True,
                 kernel_initializer='he_normal', name='gru3_b')(x)
    x = concatenate([gru_3, gru_3b])
    
    x = Dense(n_hidden, activation='relu')(x)
    # x = Dropout(0.05)(x)
    x = Dense(n_character, activation='softmax', name='base_model_out')(x)
    base_model = Model(inputs=input_tensor, outputs=x)
    plot_model(base_model, to_file="model.png", show_shapes=True)
    # base_model.summary()
    # return
    labels = Input(name='the_labels', shape=[max_labellen], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,),
                      name='ctc')([x, labels, input_length, label_length])
    model = Model(inputs=[input_tensor, labels, input_length, label_length], outputs=[loss_out])
    model.summary()
    # return
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta')
    if cont_from_checkpointfile is not None:
        savedbasemodel_fname = 'keras-ds2-ctcbase.h5'
        model.load_weights(cont_from_checkpointfile)
        base_model.save(savedbasemodel_fname, overwrite=True)
    
    checkpoint = ModelCheckpoint(filepath=os.path.join('./', 'keras-ds2-cp-weights-{epoch:02d}.h5'),
                                 save_best_only=True, save_weights_only=True)
    evaluator = Evaluate(base_model)
    model.fit_generator(gen_ctc_byclass(trainGen), steps_per_epoch=trainGen.numSteps, epochs=NB_EPOCH,
                        callbacks=[EarlyStopping(patience=5, verbose=1), checkpoint],
                        validation_data=gen_ctc_byclass(validGen), validation_steps=validGen.numSteps, verbose=VERBOSE)
    savedmodel_fname = 'keras-ds2-weights.h5'
    model.save_weights(savedmodel_fname, overwrite=True)
    savedbasemodel_fname = 'keras-ds2-ctcbase.h5'
    base_model.save(savedbasemodel_fname, overwrite=True)


allwords = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "zero", "one", "two", "three",
            "four", "five", "six", "seven", "eight", "nine", "bed", "bird", "cat", "dog", "happy", "house", "marvin",
            "sheila", "tree", "wow"]

def get_bestmatch_keywords_using_wer(str):
    global allwords
    if str in allwords:
        return str
    r = []
    str1 = ' '.join(list(str))
    for o in allwords:
        o1 = ' '.join(list(o))
        # print (type(o), type(str), o1, str1)
        r.append(wer(o1, str1))
    idx = np.argmin(np.array(r), axis=0)
    # print (idx)
    # print(str, allwords[idx])
    return allwords[idx]


def do_evaluate_CTC(savedmodel_fname):
    base_model = keras.models.load_model(savedmodel_fname)
    base_model.summary()
    testGen = dataGen_mfcc_ctc(testfile, batchSize=BATCH_SIZE * 8)
    total = 0
    diff = 0
    print('Starting evaluate...')
    start = time.time()
    for i in range(testGen.numSteps + 1):
        [X_test, y_test, _, _], _ = next(gen_ctc_byclass(testGen))

        y_true = [[alphabet._label_to_str[y] for y in x] for x in y_test]
        y_true = [''.join(x).strip() for x in y_true]
        # print (y_true)
        y_pred = base_model.predict(X_test, verbose=2)
        input_len = np.ones(y_pred.shape[0]) * y_pred.shape[1]
        # print(y_pred.shape, input_len.shape)
        y_pred = K.get_value(K.ctc_decode(y_pred, input_length=input_len)[0][0])
        y_pred = [[alphabet._label_to_str[y] for y in x if y >= 0] for x in y_pred]
        y_pred = [''.join(x).strip() for x in y_pred]
        y_pred = [get_bestmatch_keywords_using_wer(x) for x in y_pred]
        # print(y_true)
        # print(y_pred)
        idx = 0
        for a, b in list(zip(y_true, y_pred)):
            total += 1
            if a != b:
                # print('DIFF!!!', a, b)  # , y_pred_strs[idx])
                diff += 1
            idx += 1
        print('diff/total', diff, total, round(diff / total, 3))
        # break
    print('test count', testGen.totallen)
    print('time cost', time.time() - start)
    return



########################################################################
if __name__ == '__main__':
    print('Start ... ')
    
    # do_train_ds2_CTC()
    # do_train_ds2_CTC('keras-ds2-cp-weights-01.h5')

    do_evaluate_CTC('keras-ds2-ctcbase.h5')
    
    print('End!')
