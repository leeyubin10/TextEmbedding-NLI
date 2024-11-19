# -*- coding: utf-8 -*-

import random
from tqdm import tqdm
import numpy as np
import torch
from torch.optim import Adam
from sklearn.metrics import accuracy_score, f1_score

from datamodule import DownStreamDataModule, CBOWDataModule
from model import SNLIModel_CNN, CBOW
from tokenizer import SNLITokenizer
import json

def write_json(data, path):
    with open(path, 'w', encoding='UTF8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        

class CBOWTrainer:
    def __init__(self, config):
        self.config = config
        self.tokenizer = SNLITokenizer()
        self.datamodule = CBOWDataModule(self.config, self.tokenizer)
        self.model = CBOW(self.config, self.tokenizer).to('cuda')
        
        self.train_dataloader = self.datamodule.train_dataloader()
        self.val_dataloader = self.datamodule.val_dataloader()
        self.test_dataloader = self.datamodule.test_dataloader()
        
        self.optimizer = Adam(self.model.parameters(), lr = self.config.lr, eps = self.config.eps, weight_decay = self.config.weight_decay)
        
    
    def train(self):
        for epoch in tqdm(range(self.config.num_epochs)):
            self.model.zero_grad()
            self.model.train(True)

            train_loss = 0.0

            for batch in tqdm(self.train_dataloader):
                inputs = {
                    'context' : batch['context'].to('cuda'),
                    'target' : batch['target'].to('cuda'),
                }

                loss, pred = self.model(**inputs) 
                
                train_loss += loss.item()

                loss.backward()
                self.optimizer.step()
                self.model.zero_grad()

            print("|{:^79}|".format(" Epoch / Total Epoch : {} / {} ".format(epoch, self.config.num_epochs)))
            print("|{:^79}|".format(" CBOW Train Loss : {:.4f}".format(train_loss / len(self.train_dataloader))))

            self.model.train(False)

            self.valid()
            
    def valid(self):
        self.model.eval()

        val_loss = 0.0

        for batch in tqdm(self.val_dataloader):
            inputs = {
                    'context' : batch['context'].to('cuda'),
                    'target' : batch['target'].to('cuda'),
                }

            loss, pred = self.model(**inputs) 
            
            val_loss += loss.item()

        print("|{:^79}|".format(" CBOW Valid Loss : {:.4f}".format(val_loss / len(self.val_dataloader))))

    def test(self):
        self.model.eval()

        test_loss = 0.0
        test_pred = []
        test_label = []


        for batch in tqdm(self.test_dataloader):
            inputs = {
                    'context' : batch['context'].to('cuda'),
                    'target' : batch['target'].to('cuda'),
                }

            loss, pred = self.model(**inputs) 
            
            test_loss += loss.item()

            test_label.append(batch['target'].cpu().detach().numpy())
            test_pred.append(pred.cpu().detach().numpy())
        
        test_label, test_pred = np.concatenate(test_label), np.concatenate(test_pred)
        test_acc = accuracy_score(test_label, test_pred)

        write_json(test_pred.tolist(),'CBOW_test_pred.json')
        
        torch.save({'embedding.weight': self.model.embedding.weight}, self.config.save_name + '.pth')

        print("|{:^79}|".format(" CBOW Test Loss : {:.4f} | Test Accuracy : {:.4f}".format(test_loss / len(self.test_dataloader), test_acc)))
        
        

    def train_and_test(self):
        self.train()
        self.test()

class DownstreamTrainer:
    def __init__(self, config):
        self.config = config
        self.tokenizer = SNLITokenizer()
        self.datamodule = DownStreamDataModule(self.config, self.tokenizer)

        self.model = SNLIModel_CNN(self.config, self.tokenizer).to('cuda')

        self.train_dataloader = self.datamodule.train_dataloader()
        self.val_dataloader = self.datamodule.val_dataloader()
        self.test_dataloader = self.datamodule.test_dataloader()

        self.optimizer = Adam(self.model.parameters(), lr = self.config.lr, eps = self.config.eps, weight_decay = self.config.weight_decay)

    def train(self):
        for epoch in tqdm(range(self.config.num_epochs)):
            self.model.zero_grad()
            self.model.train(True)

            train_loss = 0.0


            for batch in tqdm(self.train_dataloader):
            

                inputs = {
                    'premise_feature' : batch['premise_feature'].to('cuda'),
                    'hypothesis_feature' : batch['hypothesis_feature'].to('cuda'),
                    'label' : batch['label'].to('cuda')
                }

                loss, _= self.model(**inputs) 

                train_loss += loss.item()

                loss.backward()
                self.optimizer.step()
                self.model.zero_grad()

            print("|{:^79}|".format(" Epoch / Total Epoch : {} / {} ".format(epoch, self.config.num_epochs)))
            print("|{:^79}|".format(" Train Loss : {:.4f}".format(train_loss / len(self.train_dataloader))))

            self.model.train(False)

            self.valid()

    def valid(self):
        self.model.eval()

        val_loss = 0.0

        for batch in tqdm(self.val_dataloader):
            inputs = {
                'premise_feature' : batch['premise_feature'].to('cuda'),
                'hypothesis_feature' : batch['hypothesis_feature'].to('cuda'),
                'label' : batch['label'].to('cuda')
            }

            loss, _= self.model(**inputs) 
            
            val_loss += loss.item()

        print("|{:^79}|".format(" Valid Loss : {:.4f}".format(val_loss / len(self.val_dataloader))))

    def test(self):
        self.model.eval()

        test_loss = 0.0
        test_pred = []
        test_label = []


        for batch in tqdm(self.test_dataloader):
            inputs = {
                'premise_feature' : batch['premise_feature'].to('cuda'),
                'hypothesis_feature' : batch['hypothesis_feature'].to('cuda'),
                'label' : batch['label'].to('cuda')
            }

            loss, pred = self.model(**inputs) 
            
            test_loss += loss.item()

            test_label.append(batch['label'].cpu().detach().numpy())
            test_pred.append(pred.cpu().detach().numpy())
        
        test_label, test_pred = np.concatenate(test_label), np.concatenate(test_pred)
        test_acc = accuracy_score(test_label, test_pred)

        write_json(test_pred.tolist(),'test_pred.json')

        print("|{:^79}|".format(" Test Loss : {:.4f} | Test Accuracy : {:.4f}".format(test_loss / len(self.test_dataloader), test_acc)))

    def train_and_test(self):
        self.train()
        self.test()