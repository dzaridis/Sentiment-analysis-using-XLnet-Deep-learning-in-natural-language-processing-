#import necessary libraries
import os
import time
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import plot_confusion_matrix,classification_report
from sklearn.model_selection import train_test_split


# torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
#torch.cuda.empty_cache()

# hugging face transformers
from transformers import XLNetTokenizer, XLNetForSequenceClassification, AdamW, AutoModel
from transformers.optimization import get_linear_schedule_with_warmup

# Parser
parser = argparse.ArgumentParser(description='Team 14: Text Classification')
parser.add_argument('--dataset', type=str, help='Dataset location')
parser.add_argument('--xlnet_weights', type=str, help='XLNET LARGE model weights')
parser.add_argument('--output_dir', type=str, help='Output directory')
parser.add_argument('--epochs', type=int, help='Number of training epochs', default=10)
parser.add_argument('--num_gpus', type=int, help='Number of GPUs', default=1)

args = parser.parse_args()

# Number of GPUs available. Use 0 for CPU mode.
ngpu = args.num_gpus

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
#device = torch.device("cuda")

def read_airlines(split_dir):
    df = pd.read_csv(split_dir+'/airlines_cleaned.txt')
    df=df[["text","airline_sentiment"]]
    X_train, X_test, y_train, y_test = train_test_split(df.text, df.airline_sentiment, test_size=0.2, random_state=37)
    print('# Train data samples:', X_train.shape[0])
    print('# Test data samples:', X_test.shape[0])
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    X_train_np=X_train.to_numpy()
    X_test_np=X_test.to_numpy()
    y_train_np=y_train.to_numpy()
    y_test_np=y_test.to_numpy()
    y_enc_train=[]
    for i in range (X_train_np.shape[0]):
        if y_train_np[i]=="negative" or y_train_np[i]=="neutral":
            y_enc_train.append(0)
        elif y_train_np[i]=="positive":
            y_enc_train.append(1)
    y_enc_test=[]
    for i in range (X_test_np.shape[0]):
        if y_test_np[i]=="negative" or y_test_np[i]=="neutral":
            y_enc_test.append(0)
        elif y_test_np[i]=="positive":
            y_enc_test.append(1)

    #y_enc_train=np.asarray(y_enc_train)
    #y_enc_test=np.asarray(y_enc_test)
    x_tr=[]
    for i in range (X_train_np.shape[0]):
        x_tr.append(str(X_train_np[i]))
    #x_tr=np.asarray(x_tr)
    x_ts=[]
    for i in range (X_test_np.shape[0]):
        x_ts.append(str(X_test_np[i]))
    #x_ts=np.asarray(x_ts)
    
    return x_tr, y_enc_train, x_ts, y_enc_test


def load_dataset(path):
    train_texts, train_labels, test_texts, test_labels= read_airlines(path)

    return train_texts, train_labels, test_texts, test_labels

class AirlineDataset(Dataset):
    def __init__(self,encodings,labels):
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


def train_classifier(model, epochs, optimizer, scheduler, path):
    print("\nModel training...\n")
    model.train()
    
    losses, accuracies = [], []  # for every epoch
    
    for epoch in range(epochs):
        
        batch_losses = []
        correct, total = 0, 0 # for every batch
        
        for batch in tqdm(train_loader):
            
            # zero out gradients
            optimizer.zero_grad()
            
            # get inputs and labels
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            
            predictions = outputs.logits.argmax(1)
            correct += (predictions == labels).sum().item()
            total += len(labels)
            
            # compute loss
            loss = outputs[0]
            batch_losses.append(loss.item())
            loss.backward()
            
            # update parameters
            optimizer.step()
            scheduler.step()
        
        # save model
        model.save_pretrained(path)
        
        mean_loss = torch.tensor(batch_losses).mean()
        acc = (correct/total)*100
        print('Epoch: {}/{},  Loss={:.3f},  Accuracy={:.3f}'.format(epoch+1, epochs, mean_loss.item(), acc))


def evaluate(model, loader):
    model.eval()
    total = 0
    correct = 0
    lab=[]
    pred=[]
    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            predictions = outputs.logits.argmax(1)
            try:
                for i in predictions:
                    pred.append(i)
                for j in labels:
                    lab.append(j)
            except:
                print("check what is wrong")
            correct += (predictions == labels).sum().item()
            total += len(labels)

    print(f'Accuracy on test set: {round(100*(correct/total),3)}')
    try:
        print(classification_report(lab,pred))
    except:
        print("no classification report")
total_time = time.time()

# Loading the dataset
print("Loading the dataset...\n")
timer = time.time()
train_texts, train_labels, test_texts, test_labels = load_dataset(args.dataset)
timer = time.time() - timer
print(f"Dataset is successfully loaded.\nNeeded Time: {timer}\n")


# XLNet Large 24-layer, 1024-hidden, 16-heads, 340M parameters
print("Loading the tokenizer...\n")
timer = time.time()
tokenizer = XLNetTokenizer.from_pretrained(args.xlnet_weights+'xlnet_token_class/')
timer = time.time() - timer
print(f"Tokenizer is successfully loaded.\nNeeded Time: {timer}\n")


print("Parsing the data to the tokenizer and then to dataloader...\n")
timer = time.time()
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=64)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=64)

train_dataset = AirlineDataset(train_encodings, train_labels)
test_dataset = AirlineDataset(test_encodings, test_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
timer = time.time() - timer
print(f"Dataset is successfully loaded to the dataloader.\nNeeded Time: {timer}\n")


# define model and optimizer 
print("Loading the XLNET-LARGE model...\n")
timer = time.time()
model = XLNetForSequenceClassification.from_pretrained(args.xlnet_weights+'xlnet_model_class/')
model.to(device)
timer = time.time() - timer
print(f"Model is successfully loaded.\nNeeded Time: {timer}\n")


# optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=500, num_training_steps=4000)


# Start training
try:
    os.makedir(args.output_dir)
except:
    print("The output directory already exists.\n")
timer = time.time()
train_classifier(model, epochs=args.epochs, optimizer=optimizer, scheduler=scheduler, path=args.output_dir)
print("Training finished.\n")
timer = time.time() - timer
print(f"Model is successfully trained.\nNeeded Time: {timer}\n")


# load model for evaluation
model = XLNetForSequenceClassification.from_pretrained(args.output_dir)
model.to(device)
"""
# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    model = nn.DataParallel(model, list(range(ngpu)))
"""
print("\nStarting evaluation...\n")
timer = time.time()
evaluate(model, test_loader)
timer = time.time() - timer
print(f"Model is successfully evaluated.\nNeeded Time: {timer}\n")


# Total Time
total_time = time.time() - total_time
total_time /= 60
print("Program finished in {:.2f} minutes.\n".format(total_time))
