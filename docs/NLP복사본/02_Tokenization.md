https://projector.tensorflow.org/

에서 메타데이터 벡터데이터를 주입하면 3차 그래프나옴



# Tokenization

**토큰화(Tokenization)**는 텍스트를 작은 단위로 나누는 과정을 의미한다. 텍스트를 의미 있는 작은 단위로 분할하여 처리하는 것은 자연어 처리 작업의 초기 단계 중 하나로 매우 중요하다.

토큰은 일반적으로 단어, 구두점, 숫자, 해시태그 등과 같이 텍스트의 의미를 가지는 작은 단위로 정의된다. 문자열 토큰화는 주어진 텍스트를 토큰 단위로 분할하여 리스트나 배열로 반환한다.



`nltk`는 자연언어 툴킷 라이브러리다. 라이브러리를 이요하여 간단히 구현하면

```python
from nltk.tokenize import word_tokenize
word_tokenize("get out!")
```

```
['get', 'out', '!']
```



## Tokenizer 방식 2가지

### 사전방식

* KoNLPy (Komoran, Mecab, Okt, 등등..) : 언어지식이 필요하다. 단어/형태소 사전
* 토큰단위: 알려진단어/형태소 및 이들의 결합
* vocab_size: unlimited
* 알려진 단어/형태소의 결합이라 가정. 필요시 형태소 분석(형태 변형 가능), 빈번하지 않은 단어를 사전에 등록해 두면 잘 인식
* 사전에 등록되지 않은 단어는 UNK 처리
* 미리 만들어져 있는 걸 이용

```
{'<OOV>': 1, '가': 2, '심하다': 3, '코': 4, '아버지': 5, '에': 6, '들어가신다': 7, '너무': 8, '코로나': 9, '비드': 10, '-': 11, '19': 12, '가방': 13, '방': 14, '너': 15, '무': 16, '는': 17, '나카무라': 18, '세이': 19, '불러': 20, '크게': 21, '히트': 22, '한': 23, '노래': 24, '입니다': 25}

[('너무', 'Adverb'),
 ('너무', 'Adverb'),
 ('너', 'Modifier'),
 ('무', 'Noun'),
 ('는', 'Josa'),
 ('나카무라', 'Noun'),
 ('세이', 'Noun'),
 ('코', 'Noun'),
 ('가', 'Josa'),
 ('불러', 'Verb'),
 ('크게', 'Noun'),
 ('히트', 'Noun'),
 ('한', 'Josa'),
 ('노래', 'Noun'),
 ('입니다', 'Adjective')]
```





### sub-words 방식

* 언어지식이 필요없다.
* 알려진 sub-words로 분해 ex) appear -> app + ear
* 자주 등장하는 단어를 제대로 인식할 가능성 높음
* 알려진 글자로 분해하여 UNK 의 개수 최소화
* 미리 만들어져 있지 않음. 어떤 말뭉치를 사용하냐에 따라 사전이 새로만들어짐
* 학습을 위해 text파일을 생성함.

```
코로나가 심하다
['▁코', '로', '나', '가', '▁심하다']
[1359, 29, 33, 13, 5383]

코비드-19가 심하다
['▁코', '비', '드', '-', '19', '가', '▁심하다']
[1359, 334, 277, 282, 3863, 13, 5383]

아버지가방에들어가신다
['▁아버지가', '방', '에', '들어가', '신', '다']
[6161, 618, 16, 13140, 267, 23]

아버지가 방에 들어가신다
['▁아버지가', '▁방', '에', '▁들어가', '신', '다']
[6161, 1569, 16, 3870, 267, 23]

너무너무너무는 나카무라세이코가 불러 크게 히트한 노래입니다
['▁너무너무너무', '는', '▁나카', '무라', '세', '이', '코가', '▁불러', '▁크게', '▁히트', '한', '▁노래', '입니다']
[14344, 12, 17264, 10088, 262, 10, 13095, 3392, 1846, 10169, 30, 765, 228]
```





각각의 장단점이 있으므로 목적에 맞게 사용해야한다.



## Tokenizer 방법

### Rule-based tokenization (공백 또는 구둣점으로 분리)

문제점 very big vocabulary 생성 ex) Transformer XL: 267,735  (공백과 구둣점으로 으로 구분했기 때문에)

large embedding matrix 생성 -> memory, time complexity 증가



### Subword tokenization

원칙: 

**빈번히 사용되는 단어**는 더 작은 subword로 나뉘어 지면 안된다.

**가끔사용되는 단어**는 의미 있는 subword로 나뉘어 져야한다.

교착어 (한국어 터키어 일본어 등)의 token화에 유용

Bert 104개 국어 version은 110000 vocab_size



## WPM 개요

하나의 단어를 내부단어(subword)로 통계에 기반하여 띄어쓰기로 분리.

하나의 단어는 의미 있는 여러 단어들의 조합으로 구성된 경우가 많기 때문에 단어를 여러 단어로 분리하여 보겠다는 전 처리 작업.

입력 문장에서 띄어쓰기는 언더바(_)로 치환

ex) Jet makers feud -> _J et _makers _fe ud _over 

치환하는 이유는 차후 다시 문장 복원을 위한 장치

WPM은 BPE(byte pair encoding) 알고리즘 사용

1994년에 제안된 데이터 압축 알고리즘

훈련데이터에 있는 단어들을 모든 글자(characters)또는 유니코드(unicode) 단위로 단어 집합(vocabulary)를 만들고, 가장 많이 등장하는 유니그램을 하나의 유니그램 단위로 통합



## Google SentencePiece

```bash
$pip install -q sentencepiece
```



BPE는 작은 사전에서 시작해서 더해가는 방식이었다면, 처음에 큰 사전으로 시작한다.

사전 토큰화 작업 없이 단어 분리 토큰화를 수행하므로 **언어에 종속되지 않음**

em _bed _ding => embedding

(first sub-word 외에는 _으로 시작)

```
너무너무너무는 나카무라세이코가 불러 크게 히트한 노래입니다
['▁너무너무너무', '는', '▁나카', '무라', '세', '이', '코가', '▁불러', '▁크게', '▁히트', '한', '▁노래', '입니다']
[14344, 12, 17264, 10088, 262, 10, 13095, 3392, 1846, 10169, 30, 765, 228]
```

Sentencepiece는 text 파일을 생성한다.



 작업순서

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import sentencepiece as spm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, Input, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical, plot_model

# read Csv
df = pd.read_csv('https://github.com/ironmanciti/NLP_Lecture/raw/master/data/ChatbotData.csv')
df.head()

All_texts = [] # spm tokenizer 학습용
Q_texts = [] # encoder 입력용
A_texts = [] # decoder 입력용 (정답 데이터)

for Q, A in df.iloc[:, [0, 1]].values:
    Q_texts.append(Q)
    A_texts.append(A)

All_texts = Q_texts + A_texts
print(len(Q_texts), len(A_texts))
```



데이터를 Load하면 학습데이터 리스트(encoder 입력데이터 + decoder 입력데이터), encoder 입력 리스트, decoder 입력 리스트를 생성한다.



그 후 Text file을 생성한다.

```python
with open('chatbot_qna.txt', 'w', encoding='utf-8') as f:
    for line in All_texts:
        f.write(line + '\n')
```



다음 명령어를 통해 `.model` , `.vocab`을 확장자를 가지는 파일을 생성한다.

```python
input_file = 'chatbot_qna.txt'
pad_id = 0 # <pad> token을 0으로 설정
vocab_size = 5000
prefix = 'chatbot_qna'
bos_id = 1 # begine of state
eos_id = 2 # end of state
unk_id = 3 # unknown

cmd = f'--input={input_file} \
--pad_id={pad_id} \
--bos_id={bos_id} \
--eos_id={eos_id} \
--unk_id={unk_id} \
--model_prefix={prefix} \
--vocab_size={vocab_size}'

spm.SentencePieceTrainer.Train(cmd)
```



파일생성완료 후 Sentencepiece 인스턴스를 생성하고 Load한다.

```python
sp = spm.SentencePieceProcessor()
sp.Load(f'{prefix}.model')
```

```
True
```



True를 출력했으면 테스트를 잠깐해본다.

```python
sp.DecodeIds([170, 367, 10, 129, 16, 4])
```

```
'저도 도움이 될 거예요.'
```



이전에 할 때 Keras로 할 때는 Decoder에서 Teacher Focing을 위해 데이터 전처리를 했다.

```python
'<sos>' + target_input + '<sos>'
target_input + '<eos>'
```

하지만 Sentencepiece에서는 이런 처리를 할 수 있는 메서드를 제공한다.



### 기능 확인

```python
sp.SetEncodeExtraOptions('bos:') # 문장 처음에 <s>를 추가한다 -> Decoder Input에 사용
pieces = sp.encode_as_pieces('아버지가 방에 들어가신다')
print(pieces)
ids = sp.encode_as_ids('아버지가 방에 들어가신다')
print(ids)
print(sp.DecodePieces(pieces))
print(sp.DecodeIds(ids))
```

```
['<s>', '▁아', '버', '지', '가', '▁방', '에', '▁들어가', '신', '다']
[1, 222, 2140, 15, 7, 1435, 29, 1687, 468, 57]
아버지가 방에 들어가신다
아버지가 방에 들어가신다
```



```python
sp.SetEncodeExtraOptions(':eos')     # 문장 끝에 </s>추가  --> Decoder target에 사용
pieces = sp.encode_as_pieces('아버지가 방에 들어가신다')
print(pieces)
ids = sp.encode_as_ids('아버지가 방에 들어가신다')
print(ids)
print(sp.DecodePieces(pieces))
print(sp.DecodeIds(ids))
```

```
['▁아', '버', '지', '가', '▁방', '에', '▁들어가', '신', '다', '</s>']
[222, 2140, 15, 7, 1435, 29, 1687, 468, 57, 2]
아버지가 방에 들어가신다
아버지가 방에 들어가신다
```



### Sequence 작성

```python
Q_sequences = [sp.encode_as_ids(sent) for sent in Q_texts]
sp.SetEncodeExtraOptions('bos:')  # 1로 시작
A_sequences_inputs = [sp.encode_as_ids(sent) for sent in A_texts] # decoder input용
sp.SetEncodeExtraOptions(':eos')
A_sequences_targets = [sp.encode_as_ids(sent) for sent in A_texts]
```

sequence 생성 다음 데이터 분포를 확인한다.

x축은 문장의 길이, y축은 그 문장의 Count

```python
max_len_Q = max(len(s) for s in Q_texts)
print("Target Text 의 최대 길이 :", max_len_Q)

max_len_A = max(len(s) for s in A_texts)
print("Target Text 의 최대 길이 :", max_len_A)

plt.hist([len(s) for s in All_texts]);
```

분포를 확인 한 뒤 최대길이를 잡는다,

```python
MAX_LEN = 30
```



### sequence padding

- encoder 는 thought vector 생성 목적이므로 default (pre) 로 padding
- decoder 는 teacher forcing 을 해야하므로 post 로 padding

```python
encoder_inputs = pad_sequences(Q_sequences, maxlen=MAX_LEN)
print("encoder input shape :", encoder_inputs.shape)
print("encoder_inputs[0] : ", encoder_inputs[1500])

decoder_inputs = pad_sequences(A_sequences_inputs, 
                               maxlen=MAX_LEN, padding="post")
print("\ndecoder input shape :", decoder_inputs.shape)
print("decoder_inputs[0] : ", decoder_inputs[1500])

decoder_targets = pad_sequences(A_sequences_targets, 
                                maxlen=MAX_LEN, padding="post")
print("\nencoder target shape :", decoder_targets.shape)
print("encoder_targets[0] : ", decoder_targets[1500])
```

