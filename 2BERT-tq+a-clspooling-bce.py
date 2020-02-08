import torch
if torch.cuda.is_available():       
	device = torch.device("cuda")
	print('There are %d GPU(s) available.' % torch.cuda.device_count())
	print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
	print('No GPU available, using the CPU instead.')
	device = torch.device("cpu")

import gc
import os
import sys
import numpy as np
import pandas as pd
import time
import datetime
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import transformers
from transformers import BertTokenizer, BertConfig, BertModel

### custom scripts
sys.path.append("./utility_scripts/")
from bert_embedder import (compute_input_arrays, 
						   compute_input_arrays_2s)
###
MODELS_PATH = "./models/" 
BERT_PATH = "./transformers/bert-base-uncased/"
MAX_SEQUENCE_LENGTH = 512
FILENAME = os.path.basename(__file__).split('.')[0]

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

####################################################################################################### 
# script functions
#######################################################################################################

def format_time(elapsed):
	'''
	Takes a time in seconds and returns a string hh:mm:ss
	'''
	# Round to the nearest second.
	elapsed_rounded = int(round((elapsed)))
	
	# Format as hh:mm:ss
	return str(datetime.timedelta(seconds=elapsed_rounded))

def compute_spearmanr(trues, preds):
	rhos = []
	for col_trues, col_pred in zip(trues.T, preds.T):
			rhos.append(spearmanr(col_trues, col_pred).correlation)
	return np.nanmean(rhos), rhos

def loss_function(predictions, targets):
	return torch.nn.BCELoss()(predictions, targets)
	
class BERTRegressor(torch.nn.Module):
	def __init__(self, bert_path, dropout, hidden_size, output_size1, output_size2):
		super().__init__()
		self.bert_layer1 = BertModel.from_pretrained(bert_path)
		self.bert_layer2 = BertModel.from_pretrained(bert_path)
		self.dropout_layer1 = torch.nn.Dropout(dropout)
		self.dropout_layer2 = torch.nn.Dropout(dropout)
		self.linear_layer1 = torch.nn.Linear(hidden_size, output_size1)
		self.linear_layer2 = torch.nn.Linear(hidden_size, output_size2)
		self.activation1 = torch.nn.Sigmoid()
		self.activation2 = torch.nn.Sigmoid()
	
	def forward(self, 
				input_word_ids1, input_masks1, input_segments1,
				input_word_ids2, input_masks2, input_segments2):
		x = self.bert_layer1(input_word_ids1, input_masks1, input_segments1)[0]
		x = self.dropout_layer1(x[:,0])
		x = self.linear_layer1(x)
		x = self.activation1(x)
		y = self.bert_layer2(input_word_ids2, input_masks2, input_segments2)[0]
		y = self.dropout_layer2(y[:,0])
		y = self.linear_layer2(y)
		y = self.activation2(y)
		return torch.cat([x,y], dim=0)

####################################################################################################### 
# loading data
#######################################################################################################
train = pd.read_csv("./input/train.csv")
target_columns = train.columns[11:]
train_targets = train.loc[:, target_columns]

####################################################################################################### 
# finetuning of the bert layer
#######################################################################################################
NUM_FOLDS = 5
DROPOUT = 0.2
LEARNING_RATE = 1e-5
EPOCHS = 25
BATCH_SIZE = 8

def train_epoch_bert(train_loader, model, optimizer, device, scheduler=None):
	train_loss = 0
	model.train()
	for batch in train_loader:
		input_ids1, input_masks1, input_segments1 = batch[0:3]
		input_ids2, input_masks2, input_segments2 = batch[3:6]
		train_targets = batch[6]

		input_ids1 = input_ids1.to(device)
		input_masks1 = input_masks1.to(device)
		input_segments1 = input_segments1.to(device)

		input_ids2 = input_ids2.to(device)
		input_masks2 = input_masks2.to(device)
		input_segments2 = input_segments2.to(device)

		train_targets = train_targets.to(device)

		optimizer.zero_grad()
		predictions = model(input_ids1, input_masks1, input_segments1, 
							input_ids2, input_masks2, input_segments2)
		loss = loss_function(predictions, train_targets.float())
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
		optimizer.step()
		if scheduler is not None:
			scheduler.step()
		train_loss += loss.item()
	return train_loss/len(train_loader)

def eval_epoch_bert(valid_loader, model, device):
	valid_loss = 0
	all_targets = list()
	all_predictions = list()
	model.eval()
	for batch in valid_loader:
		input_ids1, input_masks1, input_segments1 = batch[0:3]
		input_ids2, input_masks2, input_segments2 = batch[3:6]
		valid_targets = batch[6]

		input_ids1 = input_ids1.to(device)
		input_masks1 = input_masks1.to(device)
		input_segments1 = input_segments1.to(device)

		input_ids2 = input_ids2.to(device)
		input_masks2 = input_masks2.to(device)
		input_segments2 = input_segments2.to(device)

		valid_targets = valid_targets.to(device)

		with torch.no_grad():
			predictions = model(input_ids1, input_masks1, input_segments1, 
								input_ids2, input_masks2, input_segments2)
		loss = loss_function(predictions, valid_targets.float())
		valid_loss += loss.item()
		all_targets.append(valid_targets.detach().cpu().numpy())
		all_predictions.append(predictions.detach().cpu().numpy())

	targets = np.vstack(all_targets)
	predictions = np.vstack(all_predictions)
	valid_loss = valid_loss/len(valid_loader)
	valid_rho,valid_rhos = compute_spearmanr(targets, predictions)
	return valid_loss, valid_rho, valid_rhos

def train_bert(model, train_inputs, train_targets, valid_inputs, valid_targets, 
			   epochs, batch_size, device, patience=0, restore_best_state=True):
	train_dataset = TensorDataset(train_inputs[0], 
								  train_inputs[1],
								  train_inputs[2],
								  train_inputs[3],
								  train_inputs[4],
								  train_inputs[5],
								  torch.tensor(train_targets))
	train_sampler = RandomSampler(train_dataset)
	train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

	valid_dataset = TensorDataset(valid_inputs[0],
								  valid_inputs[1],
								  valid_inputs[2],
								  valid_inputs[3],
								  valid_inputs[4],
								  valid_inputs[5],
								  torch.tensor(valid_targets))
	valid_sampler = RandomSampler(valid_dataset)
	valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=batch_size)

	optimizer = torch.optim.AdamW(model.parameters(), LEARNING_RATE)

	best_model_state = model.state_dict()
	best_optimizer_state = optimizer.state_dict()
	best_rho = 0
	best_rhos = list()
	wait = 0
	init_time = time.time()
	
	for epoch in range(epochs):
		print(f" epoch: {epoch}".center(100, "-"))
		train_loss = train_epoch_bert(train_loader, model, optimizer, device)
		valid_loss,valid_rho,valid_rhos = eval_epoch_bert(valid_loader, model, device)
		elapsed_time = format_time(time.time()-init_time)
		print(f"elased time: {elapsed_time} - train_loss: {train_loss:.5f} - valid_loss: {valid_loss:.5f} - valid_rho: {valid_rho:5f}")
		# checking for early stopping
		if valid_rho <= best_rho:
			wait += 1
			if wait >= patience:
				print(f"Early stopping reached - best_rho: {best_rho}")
				if restore_best_state:
					print("Restoring model state from the end of the best epoch")
					model.load_state_dict(best_model_state)
					optimizer.load_state_dict(best_optimizer_state)
				break
		else:
			best_rho = valid_rho
			best_rhos = valid_rhos
			wait = 0
			if restore_best_state:
				best_model_state = model.state_dict()
				best_optimizer_state = optimizer.state_dict()
	return model, best_rho, best_rhos

### data tokenization
tokenizer = BertTokenizer(BERT_PATH+'vocab.txt', True)
train_inputs1 = compute_input_arrays_2s(train, "question_title", "question_body", tokenizer, 
										MAX_SEQUENCE_LENGTH, s1_max_length=50, s2_max_length=459, 
										sep_token="[unused0]")
train_inputs2 = compute_input_arrays(train, "answer", tokenizer, MAX_SEQUENCE_LENGTH)
train_inputs = train_inputs1+train_inputs2

kf_split = GroupKFold(n_splits=NUM_FOLDS).split(X=train, groups=train.question_body)
kfold_rho = list()
kfold_rhos = list()
for fold, (train_idx, valid_idx) in enumerate(kf_split):
	print(f" fold: {fold} ".center(100, "#"))
	_train_inputs = [train_inputs[i][train_idx] for i in range(6)]
	_train_targets = train_targets.loc[train_idx, :].values

	_valid_inputs = [train_inputs[i][valid_idx] for i in range(6)]
	_valid_targets = train_targets.loc[valid_idx, :].values

	model = BERTRegressor(bert_path=BERT_PATH, dropout=DROPOUT, hidden_size=768, 
						  output_size1=21, output_size2=9)
	model.cuda()
	model,best_rho,best_rhos = train_bert(model, _train_inputs, _train_targets, _valid_inputs, 
										  _valid_targets, EPOCHS, BATCH_SIZE, device, patience=2, 
										  restore_best_state=True)
	kfold_rho.append(best_rho)
	kfold_rhos.append(best_rhos)
	torch.save(model.state_dict(), MODELS_PATH + f"{FILENAME}-{fold}.pt")
	del model; torch.cuda.empty_cache(); gc.collect()
	
print(kfold_rho)
print(f"Mean kfold_rho: {np.mean(kfold_rho)}")

handler = open(f"{FILENAME}.info", "w")
handler.write(f"kfold_rho: {kfold_rho}\n")
handler.write(f"mean kfold_rho: {np.mean(kfold_rho)}\n")
for i in range(NUM_FOLDS):
	handler.write(f"kfold_rho fold{i}: {kfold_rhos[i]}\n")
handler.close()
