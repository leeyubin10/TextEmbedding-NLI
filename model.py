# -*- coding: utf-8 -*-

import torch.nn.functional as F
import torch.nn as nn
import torch


class CBOW(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        '''
            Guide
                torch.nn에서 사용 가능한 모델은
                Embedding, Linear으로 제한합니다.
        '''
        ############################################## EDIT ################################################
        
        self.embedding_dim = config.embedding_dim  # embedding dimension 설정
        self.vocab_size = tokenizer.vocab_size  # vocab size 설정
        
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)  # vocab_size와 embedding_dim으로 nn.Embedding 레이어 초기화
        self.linear = nn.Linear(self.embedding_dim, self.vocab_size)  # 입력 차원은 embedding_dim, 출력 차원은 vocab_size로 설정

        ############################################## EDIT ################################################
    
    def forward(self, context, target):
        '''
            Inputs
                context.shape = (batch_size, window_size * 2)
                target.shape = (batch_size, )
            
            Outputs
                loss.shape = (,)
                pred.shape = (batch_size,)
        '''
        loss, pred = None, None
        ############################################## EDIT ################################################
        
        embedded_context = self.embedding(context)  # context를 embedding에 통과시켜 각 단어를 임베딩
        summed_embedded_context = torch.sum(embedded_context, dim=1)  # 임베딩된 단어 벡터들을 모두 더하여 context 벡터를 구함
        pred_logits = self.linear(summed_embedded_context)  # context 벡터를 linear 레이어에 통과시켜 예측 logits를 구함
        
        loss_fn = nn.CrossEntropyLoss()  # 손실함수로 CrossEntropyLoss 사용
        loss = loss_fn(pred_logits, target)  # 예측 logits와 실제 타겟을 비교하여 손실을 계산

        pred = torch.argmax(pred_logits, dim=1)  # 가장 높은 확률을 갖는 클래스를 예측값으로 설정
       
        ############################################## EDIT ################################################
        return loss, pred


class SNLIModel_CNN(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        '''
            Guide
                torch.nn에서 사용 가능한 모델은
                Embedding, Linear, CNN 으로 제한합니다.
        '''
        ############################################## EDIT ################################################

        self.embedding_dim = config.embedding_dim  # embedding dimension 설정
        self.hidden_dim = config.hidden_dim  # CNN에서 사용할 hidden dimension 설정
        self.kernel_sizes = config.kernel_sizes  # kernel sizes 설정
        self.dropout = config.dropout  # dropout 설정
        self.vocab_size = tokenizer.vocab_size  #vocab size 설정

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)  # 입력 토큰을 임베딩 벡터로 변환하는 Embedding 레이어

        self.conv_layers = nn.ModuleList([
            nn.Conv1d(self.embedding_dim, self.hidden_dim, kernel_size) for kernel_size in self.kernel_sizes
        ])  # 다양한 kernel_sizes를 사용하는 Conv1d 레이어들을 생성하여 리스트로 관리

        self.fc = nn.Linear(len(self.kernel_sizes) * self.hidden_dim * 2, self.vocab_size)  # 다양한 kernel_sizes를 사용하여 추출된 특징을 하나로 결합하고, 어휘 크기만큼의 출력을 생성

        self.dropout_layer = nn.Dropout(self.dropout)  # dropout 레이어

        ############################################## EDIT ################################################

        if config.load_CBOW:
            print(f"loading CBOW from {config.load_name}.pth")
            CBOW_params = torch.load(config.load_name + '.pth')
            mapped_params = {'weight': CBOW_params['embedding.weight']}
            self.embedding.load_state_dict(mapped_params)

    def forward(self, premise_feature, hypothesis_feature, label):
        '''
            Inputs
                premise_feature.shape = (batch_size, max_length)
                hypothesis_feature.shape = (batch_size, max_length)
            
            Outputs
                loss.shape = (,)
                pred.shape = (batch_size,)
        '''
        ############################################## EDIT ################################################

        premise_embedded = self.embedding(premise_feature)  # Premise 임베딩
        hypothesis_embedded = self.embedding(hypothesis_feature)  # Hypothesis 임베딩

        # 각각의 임베딩에 대해 CNN 레이어 적용
        premise_conv_out = [F.relu(conv(premise_embedded.transpose(1, 2))) for conv in self.conv_layers]
        hypothesis_conv_out = [F.relu(conv(hypothesis_embedded.transpose(1, 2))) for conv in self.conv_layers]

        # Max pooling
        premise_pooled = [F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2) for conv_out in premise_conv_out]
        hypothesis_pooled = [F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2) for conv_out in hypothesis_conv_out]

        premise_concat = torch.cat(premise_pooled, dim=1)  # 각각의 풀링 결과를 하나의 벡터로 합치기
        hypothesis_concat = torch.cat(hypothesis_pooled, dim=1)

        combined = torch.cat([premise_concat, hypothesis_concat], dim=1)  # 두 벡터를 연결하여 하나의 벡터로 만들기
        combined = self.dropout_layer(combined)  # dropout 적용
        
        combined = combined.view(-1, len(self.kernel_sizes) * self.hidden_dim * 2)
        logits = self.fc(combined)  # Fully connected 레이어 통과

        loss = F.cross_entropy(logits, label)  # CrossEntropyLoss를 이용한 손실 계산

        pred = torch.argmax(logits, dim=1)  # 예측값 계산

        ############################################## EDIT ################################################

        return loss, pred
    