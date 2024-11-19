# NLP Homework 1: Stanford NLI Report  

## 1. CBOW 모델 구조 설명  
### 1-1) Vocab 구축  
- Premise와 Hypothesis를 tokenize하여 토큰화.  
- 모든 토큰을 정렬 후 정수 인코딩하여 Vocab 생성.  

### 1-2) Tokenizer 구축  
- 문장을 split하여 토큰 단위로 나눔.  
- Vocab에 정수 인코딩 값이 없으면 ‘UNK’(0)으로 처리.  

### 1-3) 데이터 전처리  
- Tokenizer로 정수 인코딩 수행.  
- Window size(2)를 기준으로 context와 target 생성.  
- Padding/Truncation으로 토큰 길이 조정(max_length 기준).  

### 1-4) CBOW 모델 학습  
- Context 단어를 임베딩 후 벡터 합산하여 대표 문맥 벡터 생성.  
- Linear Layer를 통해 중심 단어의 로그 확률 계산.  
- 실제 중심 단어와 비교하여 손실 계산 후 학습.  

---

## 2. CNN 모델 구조 설명  
### 2-1) Embedding Layer  
- 단어를 임베딩 벡터로 매핑.  

### 2-2) Convolutional Layer  
- 다양한 kernel_sizes로 특징 추출.  

### 2-3) Pooling Layer  
- Max Pooling으로 Convolutional 결과를 집계.  

### 2-4) Fully Connected Layer  
- Dropout 적용 후 분류 수행.  

### 2-5) Dropout Layer  
- Overfitting 방지를 위한 레이어.  

---

## 3. CBOW 임베딩 사용 유무에 따른 CNN 성능 평가  
### Hyperparameter Tuning  
- **seed**: 42  
- **embedding_dim**: 20  
- **num_epochs**: 50  
- **batch_size**: 512  
- **learning_rate**: 5e-4  
- **hidden_dim**: 50  
- **kernel_sizes**: [2,3,4]  
- **dropout**: 0.3  

### 실험 결과  
- **Learning Rate**: 큰 값은 성능 저하 → 점진적으로 감소.  
- **Epoch**: 적절한 값을 찾아 runtime 단축.  
- **hidden_dim**, **kernel_size**, **dropout** 등으로 성능 개선.  
