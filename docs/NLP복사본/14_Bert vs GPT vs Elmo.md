## ELMO (Embeddings from Language Model)

* Bidirectional LSTM을 이용하여 양방향으로 문맥 전체를 파악한다.



## OpenAI GPT1 (Generative Pre-Training Transformer)

* Transformer의 decoder 12개를 쌓아서 구성한다.
* decoder 를 사ㅛㅇ하므로 이전 단어들 만으로 다음 단어를 예측한다.
* forward  language model, autoregressive (자기 회귀)



## BERT (Bidirectional Encoder Representational from Transformers)

* Transformer의 encoder 12/24개를 쌓아서 구성

* MLM(Maksed Language Model), Next Sentence Prediction 으로 train





## GPT2

GPT2는 기본 구조는 GPT1와 동일

BERT를 이기기위해 GPT2는 사이즈만 무식하게 엄청 늘렸다. 

| Parameters | Layers | d_model |
| ---------- | ------ | ------- |
| 117M       | 12     | 768     |
| 345M       | 24     | 1024    |
| 762M       | 36     | 1280    |
| 1542M      | 48     | 1600    |

GPT2의 두 번째 size가 BERT의 가장 큰 model size이다.





## XLNet (Transformer_XL)

* PErmutation Language Modeling (PLM) 방식으로 training
* XLNet의 크기는 BERT-Large와 같은 24-layer, 340M parameters

