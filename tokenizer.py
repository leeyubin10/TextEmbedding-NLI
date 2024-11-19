# -*- coding: utf-8 -*-

import json, os
from typing import List, Dict

import torch
import numpy as np


class SNLITokenizer:
    def __init__(self):

        train = self._read_data()
        self.vocab = self._make_vocab(train)

        self.unk_token_id = self.vocab['UNK']
        self.pad_token_id = self.vocab['PAD']
        self.vocab_size = len(self.vocab)

    def __len__(self):
        return len(self.vocab)

    def _read_data(self):
        train_path = os.path.join('./data/snli_train.json')
        with open(train_path, 'r') as f:
            train = json.load(f)
        return train

    def _make_vocab(self, train: List[Dict]) -> Dict:
        '''
            Inputs
                train = [
                    {
                        "premise": "Students practicing yoga in a class setting.",
                        "hypothesis": "A yoga class is in progress.",
                        "label": 0
                    },
                    .
                    .
                    .
                    {
                        "premise": "Two people pose for the camera.",
                        "hypothesis": "Two people are yelling.",
                        "label": 2
                    }
                ]

            Outputs
                vocab = {
                    'UNK' : 0,
                    'PAD' : 1,
                    '!' : 2
                    .
                    .
                    .
                }

            Guide
                1. vocab 저장 시 오름차순으로 정렬.
                (예시) 
                {'UNK': 0, 'PAD': 1, '"28"': 2, '"3"': 3, '"513"': 4, '"A': 5, '"Arrivo".': 6, '"Babys"': 7, '"Bada': 8}
        '''
        vocab = {
            'UNK' : 0,
            'PAD' : 1
        }

        ############################################## EDIT ################################################
        
        all_tokens = []  # 모든 token을 담을 리스트 초기화
        
        for dic in train:  # train dataset 탐색 -> ex) print(dic) -> {"premise":Students ...~} 딕셔너리 형태 출력
            premise_tokens = self.tokenize(dic['premise'])  # tokenize 함수 활용하여 dic 내의 'premise' key가 가지고 있는 sentence를 tokenization
            hypothesis_tokens = self.tokenize(dic['hypothesis'])  # tokenize 함수 활용하여 dic 내의 'hypothesis' key가 가지고 있는 sentence를 tokenization
            all_tokens += premise_tokens + hypothesis_tokens  # tokenization이 된 token들을 all_tokens 리스트에 모두 추가
            
        all_tokens = sorted(list(set(all_tokens)))  # 오름차순 정렬
        
        for token in all_tokens:  # tokenization이 된 token들의 리스트를 순환
            if token not in vocab:  # 만약 vocab에 all_tokens 리스트에 저장된 token이 없다면
                vocab[token] = len(vocab)  # vocab에 추가 -> 현재 vocab에 있는'UNK', 'PAD' 토큰 이외의 모든 오름차순된 토큰들이 차례로 vocab에 저장됨
        
        ############################################## EDIT ################################################
        return vocab

    def tokenize(self, input: str) -> List[str]:
        '''
            Inputs
                "Students practicing yoga in a class setting."
            Outputs
                ['Students', 'practicing', 'yoga', 'in', 'a', 'class', 'setting.']
        '''
        tokens = input.split(' ')

        return tokens

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        '''
            Inputs
                "Students practicing yoga in a class setting."
            Outputs
                [1616, 10556, 15112, 7626, 1891, 4099, 11875]

            Guide
                1. 구성된 vocab을 잘 활용할것.
                2. Train 데이터로만 vocab를 구성하기 때문에, Valid와 Test 데이터의 token이 vocab에 존재하지 않을 수 있음.
                3. 이 경우 해당 token을 [UNK](unknown) token으로 처리.
        '''
        ids = None
        ############################################## EDIT ################################################
        
        ids = []  # 리스트 형태로 초기화
        
        for token in tokens:  # tokenization이 된 리스트 순환
            if token in self.vocab:  # 만약 tokens 내의 token이 vocab에 있다면
                ids.append(self.vocab[token])  # vocab에 저장된 token의 정수 인코딩 값을 ids 리스트에 추가
            else:  # 만약 tokens 내의 token이 vocab에 없다면
                ids.append(self.vocab['UNK'])  # vocab에 저장된 'UNK'의 정수 인코딩 값(=0)을 ids 리스트에 추가
        
        print(ids)  # 결과 확인용
        
        ############################################## EDIT ################################################
        return ids
    
if __name__ == '__main__':
    '''
        tokenizer.py 의 method의 self-check 를 해보려면 tokenizer.py 파일을 실행시켜보세요.
        self-check 결과가 100% 맞지 않을 수 있습니다.
    '''

    tokenizer = SNLITokenizer()
    
    tokens = tokenizer.tokenize("NLP is super FUNNY")
    if tokens == ['NLP', 'is', 'super', 'FUNNY']:
        print("Your 'tokenize' method is probably correct.")
    else:
        print("Your 'tokenize' method is probably wrong.")
    
    ids = tokenizer.convert_tokens_to_ids(tokens)
    if ids == [0, 7781, 13260, 0]:
        print("Your 'convert_tokens_to_ids' method is probably correct.")
    else:
        print("Your 'convert_tokens_to_ids' method is probably wrong.")
