#import necessary libraries
import os
import time
import argparse
from pathlib import Path
from tqdm import tqdm

# torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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

def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos","neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir=='neg' else 1)
    return texts, labels


def load_dataset(path):
    train_texts, train_labels = read_imdb_split(path+'train')
    test_texts, test_labels = read_imdb_split(path+'test')

    return train_texts, train_labels, test_texts, test_labels

class IMDbDataset(Dataset):
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
    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            predictions = outputs.logits.argmax(1)
            correct += (predictions == labels).sum().item()
            total += len(labels)
    print(f'Accuracy on test set: {round(100*(correct/total),3)}')

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

train_dataset = IMDbDataset(train_encodings, train_labels)
test_dataset = IMDbDataset(test_encodings, test_labels)

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
