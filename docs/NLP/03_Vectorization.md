# Vectorization

**문장의 벡터화(Vectorization)**란, 텍스트 데이터인 문장을 수치형 벡터로 변환하는 과정을 말한다. 텍스트 데이터를 모델에 입력하기 위해 필요한 과정이다. 벡터화를 통해 문장의 의미와 구조를 수치적으로 표현할 수 있으며, 이를 통해 텍스트 데이터를 분석하고 모델에 적용한다.

문장의 벡터화 방법에는 다양하다. 

* Count Vectorization
* TF-IDF (Term Frequency-Inverse Document Frequency)
* One-Hot Encoding
* Word Embedding



## Count Vectorization - BOW(bag-of-words)

문장에 등장하는 **단어의 빈도**를 계산하여 벡터로 표현한다. 각 단어의 등장 횟수를 요소로 가지는 벡터를 생성한다. 이 방법은 순서를 고려하지 않는다.

```python
from sklearn.feature_extraction.text import CountVectorizer

# 문장 리스트
sentences = [
    "I love to eat pizza",
    "Pizza is my favorite food and Pizza is God",
    "I enjoy eating pizza with friends"
]

# CountVectorizer 객체 생성
vectorizer = CountVectorizer()

# 문장을 벡터로 변환
vectorized_sentences = vectorizer.fit_transform(sentences)

# 벡터화된 문장 출력
print(f"단어의 수: {vectorized_sentences.shape[1] - 1}")
print(vectorized_sentences.toarray())
print(vectorizer.get_feature_names_out())
print(vectorized_sentences.toarray())
```

```
단어의 수: 13
[[0 1 0 0 0 0 0 0 0 1 0 1 1 0]
 [1 0 0 0 1 1 0 1 2 0 1 2 0 0]
 [0 0 1 1 0 0 1 0 0 0 0 1 0 1]]
['and' 'eat' 'eating' 'enjoy' 'favorite' 'food' 'friends' 'god' 'is'
 'love' 'my' 'pizza' 'to' 'with']
```



Pandas로 데이터프레임을 만들어서 한눈에 확인 가능하다.

```python
# 데이터 프레임 생성
df = pd.DataFrame(vectorized_sentences.toarray(), columns=features, index=sentences)

# 데이터 프레임 출력
print(df)
```

```
                                            and  eat  eating  enjoy  favorite   
I love to eat pizza                           0    1       0      0         0  \
Pizza is my favorite food and Pizza is God    1    0       0      0         1   
I enjoy eating pizza with friends             0    0       1      1         0   

                                            food  friends  god  is  love  my   
I love to eat pizza                            0        0    0   0     1   0  \
Pizza is my favorite food and Pizza is God     1        0    1   2     0   1   
I enjoy eating pizza with friends              0        1    0   0     0   0   

                                            pizza  to  with  
I love to eat pizza                             1   1     0  
Pizza is my favorite food and Pizza is God      2   0     0  
I enjoy eating pizza with friends               1   0     1  
```



### 문제점

BOW의 문제점은 단어의 순서를 무시하고 단어의 출현빈도만 나타내는 것이다. 이는 문장이나 문서의 의미와 문맥 정보를 잃을 수 있다. 예를들어 "good"과 "not good"은 정반대의 의미를 가지지만 BOW는 단어의 출현 빈도만 고려하기 때문에 두 문장을 동일하게 처리한다.





## TF-IDF (Term Frequency-Inverse Document Frequency):

문장에 등장하는 단어의 **상대적인 중요성**을 고려하여 벡터를 생성한다. 각 단어의 등장 횟수를 해당 문장에서의 상대적인 빈도로 정규화하고, 다른 문장들에 널리 등장하는 단어는 가중치를 낮게 설정한다.



```python
from sklearn.feature_extraction.text import TfidfVectorizer

# 문장 리스트
sentences = [
    "I love to eat pizza",
    "Pizza is my favorite food",
    "I enjoy eating pizza with friends"
]

# TfidfVectorizer 객체 생성
vectorizer = TfidfVectorizer()

# 문장을 벡터로 변환
vectorized_sentences = vectorizer.fit_transform(sentences)

# 단어 리스트
features = vectorizer.get_feature_names_out()

# 벡터화된 문장 출력
print(vectorized_sentences.toarray())
```

```
[[0.54645401 0.         0.         0.         0.         0.
  0.         0.54645401 0.         0.32274454 0.54645401 0.        ]
 [0.         0.         0.         0.47952794 0.47952794 0.
  0.47952794 0.         0.47952794 0.28321692 0.         0.        ]
 [0.         0.47952794 0.47952794 0.         0.         0.47952794
  0.         0.         0.         0.28321692 0.         0.47952794]]
```



pandas로 데이터 프레임을 나타내면

```python
# 데이터 프레임 생성
df = pd.DataFrame(vectorized_sentences.toarray(), columns=features, index=sentences)

print(df)
```

```
                                        eat    eating     enjoy  favorite   
I love to eat pizza                0.546454  0.000000  0.000000  0.000000  \
Pizza is my favorite food          0.000000  0.000000  0.000000  0.479528   
I enjoy eating pizza with friends  0.000000  0.479528  0.479528  0.000000   

                                       food   friends        is      love   
I love to eat pizza                0.000000  0.000000  0.000000  0.546454  \
Pizza is my favorite food          0.479528  0.000000  0.479528  0.000000   
I enjoy eating pizza with friends  0.000000  0.479528  0.000000  0.000000   

                                         my     pizza        to      with  
I love to eat pizza                0.000000  0.322745  0.546454  0.000000  
Pizza is my favorite food          0.479528  0.283217  0.000000  0.000000  
I enjoy eating pizza with friends  0.000000  0.283217  0.000000  0.479528  
```



### 문제점

TF-IDF는 단어의 상대적인 중요성을 고려하여 벡터화한것이다. 그래서 문서 내에서만 상대적인 빈도를 고려하기 때문에 다른 문서와 비교에는 제한적이다.





## One-Hot Encoding

원핫인코딩은 각 단어를 고유한 이진 벡터로 표현한다.

Count-Vectorization





### 문제점

벡터의 차원이 단어의 개수에 비례하여 증가하기 때문에 메모리낭비가 심할 수 있다(차원의 저주). 또한 단어의 의미나 유사성을반영하지 못하고, 단순히 단어의 존재 여부에만 집중한다. 벡터의 희소성 문제도 야기한다.



## Embedding 기법

위에서 벡터화 방법에는 한계가 명확하다. 이런 한계를 극복하기 위해 단어 임베딩(Word Embedding)기법이 등장한다.  단어 임베딩은 실수 벡터로 표현하여 단어 간의 의미적 유사성을 포착하는 방법이다. 

대표적인 방법에는 **Word2Vec, GloVe, FastText**가 사용되며, 단어의 의미와 관련성을 잘 포착할 수 있다.

