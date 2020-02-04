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
from bert_embedder import compute_input_arrays_tqa, compute_sentece_pair_embedding
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
	return np.nanmean(rhos)

def loss_function(predictions, targets):
	return torch.nn.BCELoss()(predictions, targets)

class OutputMLP(torch.nn.Module):
	def __init__(self, dropout, input_size, output_size):
		super().__init__()
		self.dropout_layer = torch.nn.Dropout(dropout)
		self.linear_layer = torch.nn.Linear(input_size, output_size)
		self.activation = torch.nn.Hardtanh(min_val=0.0, max_val=1.0)
	
	def forward(self, input_data):
		x = self.dropout_layer(input_data)
		x = self.linear_layer(x)
		return self.activation(x)

class BERTEmbedder(torch.nn.Module):
	def __init__(self, bert_path):
		super().__init__()
		self.bert_layer = BertModel.from_pretrained(bert_path)
	
	def forward(self, input_word_ids, input_masks, input_segments):
		x = self.bert_layer(input_word_ids, input_masks, input_segments)[0]
		return x[:,0]
	
class BERTRegressor(torch.nn.Module):
	def __init__(self, bert_path, dropout, hidden_size, output_size):
		super().__init__()
		self.bert_layer = BertModel.from_pretrained(bert_path)
		self.dropout_layer = torch.nn.Dropout(dropout)
		self.linear_layer = torch.nn.Linear(hidden_size, output_size)
		self.activation = torch.nn.Hardtanh(min_val=0.0, max_val=1.0)
	
	def forward(self, input_word_ids, input_masks, input_segments):
		x = self.bert_layer(input_word_ids, input_masks, input_segments)[0]
		x = self.dropout_layer(x[:,0])
		x = self.linear_layer(x)
		return self.activation(x)

####################################################################################################### 
# loading data
#######################################################################################################
train = pd.read_csv("./input/train.csv")
target_columns = list(train.columns[11:])
for column in target_columns:
    unique_values = np.sort(train[column].unique())
    n_unique_values = len(unique_values)
    replace_values = np.linspace(0., 1., n_unique_values)
    mapping = {unique_values[i]:replace_values[i] for i in range(n_unique_values)}
    train.loc[:, column] = train.loc[:, column].map(mapping)
train_targets = train.loc[:, target_columns]

model = BERTEmbedder(bert_path=BERT_PATH)
train_tqa_bert_encoded = compute_sentece_pair_embedding(train, model, device, 
														which="tqa", bert_path=BERT_PATH)
train_tqa_bert_encoded.reset_index(inplace=True)
bert_columns = train_tqa_bert_encoded.columns[1:]

####################################################################################################### 
# training of the output layer
#######################################################################################################
NUM_FOLDS = 5
DROPOUT = 0.2
LEARNING_RATE = 5e-4
EPOCHS = 100
BATCH_SIZE = 32

def train_epoch_mlp(train_loader, model, optimizer, device, scheduler=None):
	train_loss = 0
	model.train()
	for batch in train_loader:
		train_data,train_targets = batch
		train_data = train_data.to(device)
		train_targets = train_targets.to(device)

		optimizer.zero_grad()
		predictions = model(train_data)
		loss = loss_function(predictions, train_targets.float())
		loss.backward()
		optimizer.step()
		if scheduler is not None:
			scheduler.step()
		train_loss += loss.item()
	return train_loss/len(train_loader)

def eval_epoch_mlp(valid_loader, model, device):
	valid_loss = 0
	all_targets = list()
	all_predictions = list()
	model.eval()
	for batch in valid_loader:
		valid_data,valid_targets = batch
		valid_data = valid_data.to(device)
		valid_targets = valid_targets.to(device)
		with torch.no_grad():
			predictions = model(valid_data)
		loss = loss_function(predictions, valid_targets.float())
		valid_loss += loss.item()
		all_targets.append(valid_targets.detach().cpu().numpy())
		all_predictions.append(predictions.detach().cpu().numpy())

	targets = np.vstack(all_targets)
	predictions = np.vstack(all_predictions)
	valid_loss = valid_loss/len(valid_loader)
	valid_rho = compute_spearmanr(targets, predictions)
	return valid_loss, valid_rho

def train_mlp(model_class, train_data, train_targets, 
			  valid_data, valid_targets, epochs, batch_size, device,
			  patience=0, restore_best_state=True):
	model = model_class(dropout=DROPOUT, input_size=768, output_size=30)
	model.cuda()
	train_dataset = TensorDataset(torch.tensor(train_data), torch.tensor(train_targets))
	train_sampler = RandomSampler(train_dataset)
	train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
	valid_dataset = TensorDataset(torch.tensor(valid_data), torch.tensor(valid_targets))
	valid_sampler = RandomSampler(valid_dataset)
	valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=batch_size)
	optimizer = torch.optim.AdamW(model.parameters(), LEARNING_RATE)

	best_model_state = model.state_dict()
	best_optimizer_state = optimizer.state_dict()
	best_rho = 0
	wait = 0
	init_time = time.time()
	
	for epoch in range(epochs):
		print(f" epoch: {epoch}".center(100, "-"))
		train_loss = train_epoch_mlp(train_loader, model, optimizer, device)
		valid_loss,valid_rho = eval_epoch_mlp(valid_loader, model, device)
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
			wait = 0
			if restore_best_state:
				best_model_state = model.state_dict()
				best_optimizer_state = optimizer.state_dict()
	return model,best_rho

kf_split = GroupKFold(n_splits=NUM_FOLDS).split(X=train, groups=train.question_body)
kfold_rhos = list()
all_models = list()
for fold, (train_idx, valid_idx) in enumerate(kf_split):
	print(f" fold: {fold} ".center(100, "#"))
	train_inputs = train_tqa_bert_encoded.loc[train_idx, bert_columns].values
	_train_targets = train_targets.loc[train_idx, :].values
	
	valid_inputs = train_tqa_bert_encoded.loc[valid_idx, bert_columns].values
	_valid_targets = train_targets.loc[valid_idx, :].values

	model,best_rho = train_mlp(OutputMLP, train_inputs, _train_targets, 
							   valid_inputs, _valid_targets, EPOCHS, BATCH_SIZE, device,
							   patience=3, restore_best_state=True)
	all_models.append(model)
	kfold_rhos.append(best_rho)

print(kfold_rhos)
print(f"Mean kfold_rhos: {np.mean(kfold_rhos)}")

####################################################################################################### 
# finetuning of the bert layer
#######################################################################################################
DROPOUT = 0.2
LEARNING_RATE = 2e-5
EPOCHS = 10
BATCH_SIZE = 14

def train_epoch_bert(train_loader, model, optimizer, device, scheduler=None):
	train_loss = 0
	model.train()
	for batch in train_loader:
		input_ids,input_masks,input_segments,train_targets = batch
		input_ids = input_ids.to(device)
		input_masks = input_masks.to(device)
		input_segments = input_segments.to(device)
		train_targets = train_targets.to(device)

		optimizer.zero_grad()
		predictions = model(input_ids, input_masks, input_segments)
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
		input_ids,input_masks,input_segments,valid_targets = batch
		input_ids = input_ids.to(device)
		input_masks = input_masks.to(device)
		input_segments = input_segments.to(device)
		valid_targets = valid_targets.to(device)
		with torch.no_grad():
			predictions = model(input_ids, input_masks, input_segments)
		loss = loss_function(predictions, valid_targets.float())
		valid_loss += loss.item()
		all_targets.append(valid_targets.detach().cpu().numpy())
		all_predictions.append(predictions.detach().cpu().numpy())

	targets = np.vstack(all_targets)
	predictions = np.vstack(all_predictions)
	valid_loss = valid_loss/len(valid_loader)
	valid_rho = compute_spearmanr(targets, predictions)
	return valid_loss, valid_rho

def train_bert(model, train_inputs, train_targets, 
			   valid_inputs, valid_targets, epochs, batch_size, device,
			   patience=0, restore_best_state=True):
	train_dataset = TensorDataset(train_inputs[0], 
								  train_inputs[1],
								  train_inputs[2],
								  torch.tensor(train_targets))
	train_sampler = RandomSampler(train_dataset)
	train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

	valid_dataset = TensorDataset(valid_inputs[0],
								  valid_inputs[1],
								  valid_inputs[2], 
								  torch.tensor(valid_targets))
	valid_sampler = RandomSampler(valid_dataset)
	valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=batch_size)

	optimizer = torch.optim.AdamW(model.parameters(), LEARNING_RATE)

	best_model_state = model.state_dict()
	best_optimizer_state = optimizer.state_dict()
	best_rho = 0
	wait = 0
	init_time = time.time()
	
	for epoch in range(epochs):
		print(f" epoch: {epoch}".center(100, "-"))
		train_loss = train_epoch_bert(train_loader, model, optimizer, device)
		valid_loss,valid_rho = eval_epoch_bert(valid_loader, model, device)
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
			wait = 0
			if restore_best_state:
				best_model_state = model.state_dict()
				best_optimizer_state = optimizer.state_dict()
	return model,best_rho

tokenizer = BertTokenizer(BERT_PATH+'vocab.txt', True)
train_inputs = compute_input_arrays_tqa(train, tokenizer, MAX_SEQUENCE_LENGTH)

kf_split = GroupKFold(n_splits=NUM_FOLDS).split(X=train, groups=train.question_body)
kfold_rhos = list()
for fold, (train_idx, valid_idx) in enumerate(kf_split):
	print(f" fold: {fold} ".center(100, "#"))
	_train_inputs = [train_inputs[i][train_idx] for i in range(3)]
	_train_targets = train_targets.loc[train_idx, :].values

	_valid_inputs = [train_inputs[i][valid_idx] for i in range(3)]
	_valid_targets = train_targets.loc[valid_idx, :].values

	model = BERTRegressor(bert_path=BERT_PATH, dropout=DROPOUT, hidden_size=768, output_size=30)
	model.load_state_dict(all_models[fold].state_dict(), strict=False)
	model.cuda()
	model,best_rho = train_bert(model, _train_inputs, _train_targets, 
								_valid_inputs, _valid_targets, EPOCHS, BATCH_SIZE, device,
								patience=2, restore_best_state=True)
	kfold_rhos.append(best_rho)
	torch.save(model.state_dict(), MODELS_PATH + f"{FILENAME}-{fold}.pt")
	del model; torch.cuda.empty_cache(); gc.collect()
	
print(kfold_rhos)
print(f"Mean kfold_rhos: {np.mean(kfold_rhos)}")

handler = open(f"{FILENAME}.info", "w")
handler.write(f"kfold_rhos: {kfold_rhos}\n")
handler.write(f"mean kfold_rho: {np.mean(kfold_rhos)}\n")
handler.close()
