## 개체명 인식

이름을 가진 개체를 인식하는 것



## 적용분야

* Search Engine 효율화

  검색 key word 추출

* Recommendation engine

  유사한 NER history 의 사용자끼리 grouping

* Customer service

  Customer 와 service agent 간 matching

* Automatic Trading

  CEO, 회사명 등 NER 이용 news 검색 후 sentiment analysts 가 긍정적이면 매수



## RNN을 이용한 NER

many to many



## 개체명 표시 방법

BIO (Begin, Inside, Outside)로 표시

* B: 개체명이 시작되는 부분
* I: 개체명의 내부부분
* O : 개체명 아닌 부분



```
해리포터 보러 메가박스 가자
B I I I O O B I I I O O
movie..     theater..
```



나만의 데이터

```
[단어][품사 Tagging][chunk tagging][개체명 tagging]
```

```
The DT B-NP O

품사 태깅과 chunk 태깅은 언어적 지식이 필요한부분이다.
```

우리는 마지막의 개체명 tagging에 집중해야한다.





