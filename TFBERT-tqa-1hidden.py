#!/usr/bin/env python
# coding: utf-8

# tensorflow on gpu
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

import gc
import os
import sys
import numpy as np
import pandas as pd
import transformers
from transformers import BertTokenizer, BertConfig, TFBertModel
from sklearn.model_selection import GroupKFold

### custom scripts
sys.path.append("./utility_scripts/")
from ml_stratifiers import MultilabelStratifiedKFold
from custom_callbacks import EarlyStopping
from bert_embedder import compute_input_arrays_tqa, compute_sentece_pair_embedding
###

MODELS_PATH = "./models/" 
BERT_PATH = "./transformers/bert-base-uncased/"
MAX_SEQUENCE_LENGTH = 512
SEED = 19

####################################################################################################### 
# loading data
#######################################################################################################
train = pd.read_csv("./input/train.csv")
target_columns = list(train.columns[11:])
train_targets = train.loc[:, target_columns]

train_tqa_bert_encoded = compute_sentece_pair_embedding(train, which="tqa", bert_path=BERT_PATH)
train_tqa_bert_encoded.reset_index(inplace=True)
bert_columns = train_tqa_bert_encoded.columns[1:]

####################################################################################################### 
# training of the output layer
#######################################################################################################
SEED = 19
NUM_FOLDS = 5
DROPOUT = 0.2
ACTIVATION = "sigmoid"
LEARNING_RATE = 5e-4
EPOCHS = 100
BATCH_SIZE = 32

def get_output_model(input_size, output_size, activation, dropout):
    input_layer = tf.keras.layers.Input((input_size,), dtype=tf.float32, name='input')
    input_layer_dpout = tf.keras.layers.Dropout(dropout)(input_layer)
    output_layer = tf.keras.layers.Dense(output_size, 
                                         activation=activation, 
                                         name="output")(input_layer_dpout)
    model = tf.keras.models.Model(inputs=input_layer,
                                  outputs=output_layer)
    return model

kf = MultilabelStratifiedKFold(n_splits=NUM_FOLDS, random_state=SEED, shuffle=True)
kf_split = kf.split(train ,train.loc[:, target_columns])
output_models = list()
kfold_scores = list()
for fold, (train_idx, valid_idx) in enumerate(kf_split):
    print(f" fold {fold} ".center(120, "#"))
    model = get_output_model(input_size=768, 
                             output_size=30,
                             activation=ACTIVATION,
                             dropout=DROPOUT)
        
    train_inputs = train_tqa_bert_encoded.loc[train_idx, bert_columns].values
    _train_targets = train_targets.loc[train_idx, :].values
    
    valid_inputs = train_tqa_bert_encoded.loc[valid_idx, bert_columns].values
    _valid_targets = train_targets.loc[valid_idx, :].values
       
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(loss="mse", optimizer=optimizer)
    callback = EarlyStopping(validation_data=(valid_inputs, _valid_targets),
                             batch_size=BATCH_SIZE,
                             patience=3,
                             restore_best_weights=True,
                             mode='max',
                             verbose=1)
    model.fit(train_inputs, _train_targets, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True,
              validation_data=(valid_inputs, _valid_targets),
              callbacks=[callback])
    output_models.append(model)
    kfold_scores.append(callback.best)

print(kfold_scores)
print(f"Mean k-fold rho: {np.mean(kfold_scores)}")

# saving the output layer models
for fold,model in enumerate(output_models):
    model.save(MODELS_PATH + f"output_tqa_1h_fold{fold}.h5")

####################################################################################################### 
# finetuning of the bert layer
#######################################################################################################
SEED = 19
NUM_FOLDS = 5
DROPOUT = 0.1
LEARNING_RATE = 2e-5
EPOCHS = 10
BATCH_SIZE = 12

def get_model(output_model, dropout=0.2, output_layer_name="output"):
    input_word_ids = tf.keras.layers.Input(
        (MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_word_ids')
    input_masks = tf.keras.layers.Input(
        (MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_masks')
    input_segments = tf.keras.layers.Input(
        (MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_segments')

    config = BertConfig()
    bert_layer = TFBertModel.from_pretrained(BERT_PATH, config=config)
    hidden_layer,_ = bert_layer([input_word_ids, input_masks, input_segments])

    hidden_layer_cls = tf.reshape(hidden_layer[:,0], (-1,768))
    
    hidden_layer_dpout = tf.keras.layers.Dropout(dropout)(hidden_layer_cls)
    output_layer = output_model.get_layer(output_layer_name)(hidden_layer_dpout)
    model = tf.keras.models.Model(
        inputs=[input_word_ids, input_masks, input_segments], 
        outputs=output_layer)
    return model

tokenizer = BertTokenizer(BERT_PATH+'vocab.txt', True)
train_inputs = compute_input_arrays_tqa(train, tokenizer, MAX_SEQUENCE_LENGTH)

kf = MultilabelStratifiedKFold(n_splits=NUM_FOLDS, random_state=SEED, shuffle=True)
kf_split = kf.split(train ,train.loc[:, target_columns])

kfold_scores = list()
for fold, (train_idx, valid_idx) in enumerate(kf_split):
  print("#"*120)
  print(f" fold {fold} ".center(120, "#"))
  print("#"*120)
  model = get_model(output_model = output_models[fold],
                    dropout = DROPOUT,
                    output_layer_name = "output")

  _train_inputs = [train_inputs[i][train_idx] for i in range(3)]
  _train_targets = train_targets.loc[train_idx, :].values

  _valid_inputs = [train_inputs[i][valid_idx] for i in range(3)]
  _valid_targets = train_targets.loc[valid_idx, :].values
      
  optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
  model.compile(loss="mse", optimizer=optimizer)
  callback = EarlyStopping(validation_data=(_valid_inputs, _valid_targets),
                            batch_size=BATCH_SIZE,
                            patience=2,
                            restore_best_weights=True,
                            mode='max',
                            verbose=1)
  model.fit(_train_inputs, _train_targets, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True,
            validation_data=(_valid_inputs, _valid_targets),
            callbacks=[callback])
  kfold_scores.append(callback.best)
  model.save(MODELS_PATH + f"bert_tqa_1h_fold{fold}.h5")
  del model
  gc.collect()

print(f"k-fold rho values: {kfold_scores}")
print(f"mean rho value: {np.mean(kfold_scores)}")
