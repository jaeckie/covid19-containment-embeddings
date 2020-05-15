# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 18:30:55 2020

@author: moder
"""
import re
import pandas as pd
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn. functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

CONTEXT_SIZE = 2
EMBEDDING_DIM = 30
LEARNING_RATE = 5e-3
EPOCHS = 20
BATCH_SIZE = 128

DATA_DICT = {'Sweden' : r'data/wikipedia_text/20200405_155601_Sweden_covid19-wikipedia.txt',
             'China' : r'data/wikipedia_text/20200405_155626_China_covid19-wikipedia.txt',
             'Austria': r'data/wikipedia_text/20200405_155727_Austria_covid19-wikipedia.txt',
             'Australia': r'data/wikipedia_text/20200405_155702_Australia_covid19-wikipedia.txt',
             'Denmark': r'data/wikipedia_text/20200405_155738_Denmark_covid19-wikipedia.txt',
             'Italy': r'data/wikipedia_text/20200405_155726_Italy_covid19-wikipedia.txt',
             'Spain': r'data/wikipedia_text/20200405_155628_Spain_covid19-wikipedia.txt',
             'Switzerland': r'data/wikipedia_text/20200405_155658_Switzerland_covid19-wikipedia.txt',
             'Greece': r'data/wikipedia_text/20200405_155742_Greece_covid19-wikipedia.txt',
             'Russia': r'data/wikipedia_text/20200405_155705_Russia_covid19-wikipedia.txt',
             'Slovenia': r'data/wikipedia_text/20200405_155657_Slovenia_covid19-wikipedia.txt',
             'Iran': r'data/wikipedia_text/20200405_155708_Iran_covid19-wikipedia.txt'}

class CBOW(nn.Module):
    
    def __init__(self, vocab_size, context_size, embedding_dim):
        super(CBOW, self).__init__()
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.embeds = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(2*context_size*embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)
        
    def forward(self, inputs):
        embeds = self.embeds(inputs).view((-1, self.context_size * 2 * self.embedding_dim)) #batch size, 2*context_size*embedding_dim
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs
    
    def save_embeddings(word_to_ix, filepath): #word_to_ix
        ems = self.embeds.weight.data.numpy()
        words = list(word_to_ix.keys())
        assert len(words) == len(ems)
        with open(filepath, 'w', encoding='utf-8') as file:
            for i in range(len(ems)):
                file.writelines('%s %s' % (words[i], ems[i,:]))
        file.close()
        print('Embeddings stored in %s...' % filepath)
    
def extract_text(data_dict):
    text = ''
    for key, val in data_dict.items():
        print('Reading data of %s...' % key)
        file = open(val, 'r', encoding='utf-8')  
        text += ' '
        text += file.read()
        file.close()
              
    text = text.lower()    
    text = text.replace('.', ' ')
    text = text.replace(',', ' ')  
    text = text.replace(';', ' ') 
    text = text.replace(':', ' ') 
    text = text.replace('"', ' ')
    text = text.replace('(', ' ') 
    text = text.replace(')', ' ') 
    text = text.replace('/', ' ') 
    # lowercase, remove brackets, remove " 
    # replace references [.+]
    text = re.sub(r'\[\d+\]', ' ', text) 
    text = text.replace('[', ' ') 
    text = text.replace(']', ' ') 
    return text.split()
   
    
def cbow_tuple(raw_text, context_size):
    raw_text_size = len(raw_text)
    data = []
    for i in range(context_size, raw_text_size-context_size):
        target = raw_text[i]
        context = raw_text[(i-context_size):i] + raw_text[(i+1):(i+context_size+1)]
        data.append((context, target))     
    return data

if __name__ == '__main__':
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device %s will be used ...' % device)
    torch.cuda.set_device(device)
    time = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # raw_text_size = len(raw_text)
    # vocab = set(raw_text)
    text = extract_text(DATA_DICT)  
    data = cbow_tuple(text, CONTEXT_SIZE)
    
    vocab = set(text) # clones text, removes duplicate words
    vocab_size = len(vocab)
    word_to_ix = {word : i for i, word in enumerate(vocab) }
    
    losses = []
    loss_function = nn.NLLLoss()
    model = CBOW(vocab_size, CONTEXT_SIZE, EMBEDDING_DIM).cuda()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    
    # init tensorboard writer
    foldername = '%s_covid19-containment-embeds_lr=%1.4f_batch_size=%d_vocabs=%d_embeds=%d' % (time, LEARNING_RATE, BATCH_SIZE, vocab_size, EMBEDDING_DIM)
    writer = SummaryWriter('runs/%s' % foldername)
    writer.add_embedding(model.embeds.weight, metadata=vocab, global_step=-1, 
                         tag='default', metadata_header=None)
    
    print('Tensorboard SummaryWriter created in %s ...' % foldername)
    
    for epoch in range(EPOCHS):
        total_loss = 0
        rand_idxs = torch.randperm(len(data))
        #for context, target in data:
        for i in range(0, len(data), BATCH_SIZE):
            # 1. init
            model.zero_grad()
            
            # 2. data
            # prepare data batch
            context_batch_list = [data[idx][0] for idx in rand_idxs[i:i+BATCH_SIZE]] # tuple, first entry
            target_batch = [data[idx][1] for idx in rand_idxs[i:i+BATCH_SIZE]] # tuple, second entry
            
            context_idxs_batch = np.zeros((len(context_batch_list), 2*CONTEXT_SIZE))
            for j, item in enumerate(context_batch_list):
                context_idxs = [word_to_ix[w] for w in item]
                context_idxs_batch[j, :] = np.array(context_idxs)  
                
            target_idxs = [word_to_ix[w] for w in target_batch]        

            # 3. foward pass
            log_probs = model(torch.tensor(context_idxs_batch, dtype=torch.long, 
                                       device=device))
            # 4. loss
            loss = loss_function(log_probs, torch.tensor(target_idxs, dtype=torch.long, #[target_ix]
                                                 device=device))
            # 5. update gradients
            loss.backward()
            optimizer.step()
            # keep track of loss
            total_loss += loss.item()
            
        # if (epoch % 5) == 0: 
        print('Epoch %d, loss: %4.2f ...' % (epoch, total_loss))
        writer.add_scalar('training loss', total_loss/1000, epoch)
        # store current embeddings  
        writer.add_embedding(model.embeds.weight, metadata=vocab, global_step=epoch, tag='default')
        #model.save_embeddings(word_to_ix, ('embeds_epoch=%d.txt' % epoch))
    
    writer.close()
    torch.save(model, 'pytorch-model_%s.pt' % foldername)
    #model.store()
    #process_embedding = model.embeds(torch.tensor(word_to_ix['process'], dtype=torch.long, device=device))
    #processes_embedding = model.embeds(torch.tensor(word_to_ix['processes'], dtype=torch.long, device=device))
    




