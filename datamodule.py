# -*- coding: utf-8 -*-

import json, os
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from typing import List, Tuple


class CBOWDataset(Dataset):
    def __init__(self, config, tokenizer, mode='train'):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config

        data = self._read_data(mode)
        self.data = self.context_target(data)
    
    def _read_data(self, mode):
        data_path = os.path.join('./data', 'snli_' + mode + '.json')
        with open(data_path, 'r') as f:
            data = json.load(f)
        return data
    
    def _get_context_target(self, input_sentence: str) -> Tuple[List[List[int]], List[int]]:
        '''
            Guide
                1. tokenizer.py에서 구현한 Tokenizer 활용, input_ids 획득
                2. i 번째 단어를 중심 단어(target)라고 했을 때, i 주변의 window_size * 2 만큼의 단어들이 문맥 단어(context)가 됨
                3. 문맥 단어가 문장의 시작과 끝을 벗어난 경우에는 데이터에 포함시키지 않을 것
                4. window_size의 크기에 따라 성능의 변화가 있을 수 있음
            
            Example
                Inputs 'I am a good student.' -> input_ids = [0, 1, 2, 3, 4, 5]
                Outputs (window_size = 2일 경우)
                    context = [[0, 1, 3, 4], [1, 2, 4, 5]]
                    target = [2, 3]
        '''
        context, target = None, None
        ############################################## EDIT ################################################
        
        context, target = [], []  # 리스트 형태로 초기화
        
        input_word = self.tokenizer.tokenize(input_sentence)  # tokenize 함수 실행
        input_ids = self.tokenizer.convert_tokens_to_ids(input_word)  # 정수 인코딩 함수 실행
        
        window_size = self.config.window_size  # 윈도우 사이즈 2로 지정

        for i, target_ids in enumerate(input_ids):  # ex) print(i, target_ids) -> 0 0(I), 1 1(am) -> index, item 출력
            left = input_ids[max(0, i-window_size):i]  # 문맥 단어 중 중심 단어 기준 왼쪽
            right = input_ids[i+1:i+1+window_size]  # 문맥 단어 중 중심 단어 기준 오른쪽
            
            if len(left+right) == window_size * 2:  # window_size * 2 만큼의 단어들이 문맥 단어
                context.append(left+right)  # context에 문맥 단어 추가
                target.append(target_ids)  # target에 중심 단어 추가

        ############################################## EDIT ################################################
        return context, target
    
    
    def context_target(self, data):
        output = []

        for dp in data:
            contexts, targets = self._get_context_target(dp['premise'])
            for context, target in zip(contexts, targets):
                temp = {}
                temp['context'] = context
                temp['target'] = target
                output.append(temp)

        return output
    
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):        
        output = {
            'context': torch.tensor(self.data[i]['context'], dtype=torch.long),
            'target': torch.tensor(self.data[i]['target'], dtype=torch.long)
        }

        return output


class SNLIDataset(Dataset):
    def __init__(self, config, tokenizer, mode = 'train'):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config

        data = self._read_data(mode)
        self.data = self._data_processing(data)

    def _read_data(self, mode):
        data_path = os.path.join('./data', 'snli_'+mode + '.json')
        with open(data_path, 'r') as f:
            data = json.load(f)
        return data

    def _convert_to_feature(self, input_sentence: str) -> List[int]:
        '''
            Inputs
                "Students practicing yoga in a class setting."
            Outputs
                [1616, 10556, 15112, 7626, 1891, 4099, 11875]
                [1, 1, 1, 1616, 10556, 15112, 7626, 1891, 4099, 11875]
            Guide
                1. tokenizer.py에서 구현한 Tokenizer 활용.
                2. 원활한 학습을 위해, config에 지정한 길이에 맞춰 truncation or padding 진행.
                3. truncation / padding의 앞 or 뒤 수행 위치에 따라 성능차이가 있을 수 있음.
                4. 위의 예시는 입력 token의 길이가 7, max_length가 10일때 padding을 앞에서 수행한 것. 
        '''
        feature = None

        ############################################## EDIT ################################################

        feature = []  # 리스트 형태로 초기화
        
        input_word = self.tokenizer.tokenize(input_sentence)  # tokenize 함수 실행
        input_ids = self.tokenizer.convert_tokens_to_ids(input_word)  # 정수 인코딩 함수 실행
        
        max_length = self.config.max_length  # max_length는 config에서 지정
            
        if len(input_ids) < max_length:  # 입력된 문장의 길이가 max_length보다 작을 때
            padding = [self.tokenizer.pad_token_id] * (max_length - len(input_ids))  # 'PAD' : 1 -> max_length에서 현재 문장 길이를 뺀 나머지는 padding 적용
            feature = padding + input_ids  # 입력 문장 앞에 패딩 적용
        else:  # 입력된 문장의 길이가 max_length보다 클 때
            feature = input_ids[:max_length]  # max_length보다 큰 부분을 자름

        ############################################## EDIT ################################################

        assert len(feature) == self.config.max_length

        return feature

    def _data_processing(self, data):

        output = []

        for dp in data:
            temp = {}

            temp['premise_feature'] = self._convert_to_feature(dp['premise'])
            temp['hypothesis_feature'] = self._convert_to_feature(dp['hypothesis'])
            temp['label'] = dp['label']

            """
            0 : entailment
            1 : neutral
            2 : contradiction
            """

            output.append(temp)

        return output

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        output = {
            'premise_feature' : torch.tensor(self.data[i]['premise_feature'], dtype = torch.long),
            'hypothesis_feature' : torch.tensor(self.data[i]['hypothesis_feature'], dtype = torch.long),    
            'label' : torch.tensor(self.data[i]['label'], dtype = torch.long),
        }

        return output


class CBOWDataModule:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
    
    def train_dataloader(self):
        train_dataset = CBOWDataset(self.config, self.tokenizer, mode='train')
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size = self.config.train_batch_size, sampler=train_sampler)

        return train_dataloader

    def val_dataloader(self):
        val_dataset = CBOWDataset(self.config, self.tokenizer, mode='val')
        val_sampler = SequentialSampler(val_dataset)
        val_dataloader = DataLoader(val_dataset, batch_size = self.config.val_batch_size, sampler=val_sampler)

        return val_dataloader

    def test_dataloader(self):
        test_dataset = CBOWDataset(self.config, self.tokenizer, mode='test')
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, batch_size = self.config.test_batch_size, sampler=test_sampler)

        return test_dataloader


class DownStreamDataModule:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def train_dataloader(self):
        train_dataset = SNLIDataset(self.config, self.tokenizer, mode='train')
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size = self.config.train_batch_size, sampler=train_sampler)

        return train_dataloader

    def val_dataloader(self):
        val_dataset = SNLIDataset(self.config, self.tokenizer, mode='val')
        val_sampler = SequentialSampler(val_dataset)
        val_dataloader = DataLoader(val_dataset, batch_size = self.config.val_batch_size, sampler=val_sampler)

        return val_dataloader

    def test_dataloader(self):
        test_dataset = SNLIDataset(self.config, self.tokenizer, mode='test')
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, batch_size = self.config.test_batch_size, sampler=test_sampler)

        return test_dataloader
    
    