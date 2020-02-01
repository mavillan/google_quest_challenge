import numpy as np 
import pandas as pd 
import math
from tqdm import tqdm
import tensorflow as tf
from transformers import BertTokenizer, BertConfig, TFBertModel

BERT_PATH = "/kaggle/input/transformers/bert-base-uncased/"
MAX_SEQUENCE_LENGTH = 512
# number of hidden layer of bert to use
N_HIDDEN_LAYERS = 1

##################################################################################################
# functions to compute input arrays in bert format for a sentence pair: question & answer
##################################################################################################

def get_masks_qa(tokens, max_seq_length):
    """Mask for padding"""
    n_tokens = len(tokens)
    return [1]*len(tokens) + [0]*(max_seq_length - n_tokens)

def get_segments_qa(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""  
    n_tokens = len(tokens)
    first_sep_index = tokens.index("[SEP]")
    segments = [0]*(first_sep_index+1) + [1]*(n_tokens-first_sep_index-1) 
    return segments + [0]*(max_seq_length - n_tokens)

def get_ids_qa(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids

def trim_input_qa(question_tokens, answer_tokens, max_sequence_length, 
                          question_max_len=254, answer_max_len=255):
    question_len = len(question_tokens)
    answer_len = len(answer_tokens)

    if (question_len+answer_len+3) > max_sequence_length:
        if question_len < question_max_len:
            question_new_len = question_len
            answer_new_len = answer_max_len + (question_max_len - question_len)
        elif answer_len < answer_max_len:
            answer_new_len = answer_len
            question_new_len = question_max_len + (answer_max_len - answer_len)
        else:
            question_new_len = question_max_len
            answer_new_len = answer_max_len
            
        if question_new_len+answer_new_len+3 != max_sequence_length:
            print(len(question_tokens))
            print(len(answer_tokens))
            print(question_new_len)
            print(answer_new_len)
            raise ValueError(f"New sequence length should be {max_sequence_length} but is {question_new_len+answer_new_len+3}")
        
        question_tokens = question_tokens[:question_new_len]
        answer_tokens = answer_tokens[:answer_new_len]
    
    return question_tokens, answer_tokens

def get_bert_inputs_qa(question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for BERT"""
    question_tokens = tokenizer.tokenize(question)
    answer_tokens = tokenizer.tokenize(answer)
    question_tokens,answer_tokens = trim_input_qa(question_tokens, 
                                                  answer_tokens,
                                                  max_sequence_length)
    all_tokens = ["[CLS]"] + question_tokens  + ["[SEP]"] + answer_tokens + ["[SEP]"]
    input_ids = get_ids_qa(all_tokens, tokenizer, max_sequence_length)
    input_masks = get_masks_qa(all_tokens, max_sequence_length)
    input_segments = get_segments_qa(all_tokens, max_sequence_length)
    return [input_ids, input_masks, input_segments]

def compute_input_arrays_qa(data, tokenizer, max_sequence_length):
    input_ids, input_masks, input_segments = [], [], []
    for _,row in tqdm(data.iterrows()):
        ids,masks,segments = get_bert_inputs_qa(row.question_body, 
                                                row.answer, 
                                                tokenizer, 
                                                max_sequence_length)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)
        
    return [np.asarray(input_ids, dtype=np.int32), 
            np.asarray(input_masks, dtype=np.int32), 
            np.asarray(input_segments, dtype=np.int32)]

##################################################################################################
# functions to compute input arrays in bert format for a sentence pair: title + question & answer
##################################################################################################

def get_ids_tqa(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids

def get_masks_tqa(tokens, max_seq_length):
    """Mask for padding"""
    n_tokens = len(tokens)
    return [1]*len(tokens) + [0]*(max_seq_length - n_tokens)

def get_segments_tqa(tokens, max_seq_length):
    """Segments: 0 for the title+question, 1 for the answer"""  
    n_tokens = len(tokens)
    first_sep_index = tokens.index("[SEP]")
    segments = [0]*(first_sep_index+1) + [1]*(n_tokens-first_sep_index-1) 
    return segments + [0]*(max_seq_length - n_tokens)

def trim_input_tqa(title_tokens, question_tokens, answer_tokens, 
                   max_sequence_length, title_max_len=30, question_max_len=239, answer_max_len=239):
    title_len = len(title_tokens)
    question_len = len(question_tokens)
    answer_len = len(answer_tokens)

    if (title_len + question_len + answer_len + 4) > max_sequence_length:

        if title_len < title_max_len:
            title_new_len = title_len
            question_max_len = question_max_len + math.ceil((title_max_len-title_len)/2)
            answer_max_len = answer_max_len + math.floor((title_max_len-title_len)/2)
        else:
            title_new_len = title_max_len

        if question_len < question_max_len:
            question_new_len = question_len
            answer_new_len = answer_max_len + (question_max_len - question_len)
        elif answer_len < answer_max_len:
            answer_new_len = answer_len
            question_new_len = question_max_len + (answer_max_len - answer_len)
        else:
            question_new_len = question_max_len
            answer_new_len = answer_max_len
            
        if title_new_len + question_new_len + answer_new_len + 4 != max_sequence_length:
            raise ValueError(f"New sequence length should be {max_sequence_length} but is {question_new_len+answer_new_len+4}")
        
        title_tokens = title_tokens[:title_new_len]
        question_tokens = question_tokens[:question_new_len]
        answer_tokens = answer_tokens[:answer_new_len]
    
    return title_tokens,question_tokens,answer_tokens


def get_bert_inputs_tqa(title, question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for BERT"""
    title_tokens = tokenizer.tokenize(title)
    question_tokens = tokenizer.tokenize(question)
    answer_tokens = tokenizer.tokenize(answer)
    title_tokens,question_tokens,answer_tokens = trim_input_tqa(title_tokens,
                                                                question_tokens, 
                                                                answer_tokens,
                                                                max_sequence_length)
    all_tokens = ["[CLS]"] + title_tokens + ["[SPECIAL_TOKEN]"] + question_tokens  + ["[SEP]"] + answer_tokens + ["[SEP]"]
    input_ids = get_ids_tqa(all_tokens, tokenizer, max_sequence_length)
    input_masks = get_masks_tqa(all_tokens, max_sequence_length)
    input_segments = get_segments_tqa(all_tokens, max_sequence_length)
    return [input_ids, input_masks, input_segments]

def compute_input_arrays_tqa(data, tokenizer, max_sequence_length):
    input_ids, input_masks, input_segments = [], [], []
    for _,row in tqdm(data.iterrows()):
        ids,masks,segments = get_bert_inputs_tqa(row.question_title,
                                                 row.question_body, 
                                                 row.answer, 
                                                 tokenizer, 
                                                 max_sequence_length)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)
        
    return [np.asarray(input_ids, dtype=np.int32), 
            np.asarray(input_masks, dtype=np.int32), 
            np.asarray(input_segments, dtype=np.int32)]

##################################################################################################
# functions to compute input arrays in bert format for a single sentence
##################################################################################################

def get_masks(tokens, max_seq_length):
    """Mask for padding"""
    n_tokens = len(tokens)
    return [1]*len(tokens) + [0]*(max_seq_length - n_tokens)

def get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    return [0]*max_seq_length

def get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab""" 
    n_tokens = len(tokens)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0]*(max_seq_length - n_tokens)
    return input_ids

def trim_input(sentence_tokens, max_sequence_length):
    if len(sentence_tokens)+2 > max_sequence_length:
        new_sentence_len = max_sequence_length-2
        sentence_tokens = sentence_tokens[:new_sentence_len]    
    return sentence_tokens

def get_bert_inputs(sentence, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for BERT"""
    sentence_tokens = tokenizer.tokenize(sentence)
    sentence_tokens = trim_input(sentence_tokens,
                                 max_sequence_length)
    all_tokens = ["[CLS]"] + sentence_tokens  + ["[SEP]"] 
    input_ids = get_ids(all_tokens, tokenizer, max_sequence_length)
    input_masks = get_masks(all_tokens, max_sequence_length)
    input_segments = get_segments(all_tokens, max_sequence_length)
    return [input_ids, input_masks, input_segments]

def compute_input_arrays(data, column, tokenizer, max_sequence_length):
    input_ids, input_masks, input_segments = [], [], []
    for _,row in tqdm(data.iterrows()):
        ids,masks,segments = get_bert_inputs(row[column],
                                             tokenizer, 
                                             max_sequence_length)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)
        
    return [np.asarray(input_ids, dtype=np.int32), 
            np.asarray(input_masks, dtype=np.int32), 
            np.asarray(input_segments, dtype=np.int32)]

##################################################################################################
# bert instantiator
##################################################################################################

def get_bert_hidden(n_hidden_layers=1, bert_path=BERT_PATH):
    input_word_ids = tf.keras.layers.Input(
        (MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_word_ids')
    input_masks = tf.keras.layers.Input(
        (MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_masks')
    input_segments = tf.keras.layers.Input(
        (MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_segments')
    
    config = BertConfig()
    config.output_hidden_states = True
    bert_layer = TFBertModel.from_pretrained(bert_path, config=config)
    hidden_layers = bert_layer([input_word_ids, input_masks, input_segments])[-1]  
    
    selected_hidden_layers = list()
    for i in range(n_hidden_layers):
        layer_idx = -(i+1)
        selected_hidden_layers.append(
            tf.reshape(hidden_layers[layer_idx][:,0], (-1,768))
        )
    if n_hidden_layers > 1:
        output_layer = tf.keras.layers.concatenate(inputs=selected_hidden_layers, axis=1)
    else:
        output_layer = selected_hidden_layers[0]
    
    bert_model = tf.keras.models.Model(inputs=[input_word_ids, input_masks, input_segments], 
                                       outputs=output_layer)
    return bert_model

def get_bert_output(bert_path=BERT_PATH):
    input_word_ids = tf.keras.layers.Input(
        (MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_word_ids')
    input_masks = tf.keras.layers.Input(
        (MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_masks')
    input_segments = tf.keras.layers.Input(
        (MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_segments')
    
    bert_layer = TFBertModel.from_pretrained(bert_path)
    _,sequence_output = bert_layer([input_word_ids, input_masks, input_segments])
        
    bert_model = tf.keras.models.Model(inputs=[input_word_ids, input_masks, input_segments], 
                                       outputs=sequence_output)
    return bert_model

##################################################################################################
# main functions
##################################################################################################

def compute_sentece_pair_embedding(data, which="qa", n_hidden_layers=N_HIDDEN_LAYERS, 
                                   bert_path=BERT_PATH):
    tokenizer = BertTokenizer(bert_path+'vocab.txt', True)
    if which == "qa":
        data_inputs = compute_input_arrays_qa(data, tokenizer, MAX_SEQUENCE_LENGTH)
    elif which == "tqa":
        data_inputs = compute_input_arrays_tqa(data, tokenizer, MAX_SEQUENCE_LENGTH)
    bert_model = get_bert_hidden(n_hidden_layers, bert_path=bert_path)
    data_bert_encoded = bert_model.predict(data_inputs, batch_size=8, verbose=1)
    column_names = list()
    for i in range(n_hidden_layers):
        layer_idx = 12-i
        column_names.extend([f"h{layer_idx}_{h}" for h in range(1, 769)])
    data_df_bert_encoded = pd.DataFrame(data_bert_encoded,
                                        columns=column_names,
                                        index=data.qa_id)
    return data_df_bert_encoded

def compute_sentence_embedding(data, column, n_hidden_layers=N_HIDDEN_LAYERS, bert_path=BERT_PATH):
    tokenizer = BertTokenizer(bert_path+'vocab.txt', True)
    data_inputs = compute_input_arrays(data, column, tokenizer, MAX_SEQUENCE_LENGTH)
    bert_model = get_bert_hidden(n_hidden_layers, bert_path=bert_path)
    data_bert_encoded = bert_model.predict(data_inputs, batch_size=8, verbose=1)
    column_names = list()
    for i in range(n_hidden_layers):
        layer_idx = 12-i
        column_names.extend([f"h{layer_idx}_{h}" for h in range(1, 769)])
    data_df_bert_encoded = pd.DataFrame(data_bert_encoded,
                                        columns=column_names,
                                        index=data.qa_id)
    return data_df_bert_encoded
