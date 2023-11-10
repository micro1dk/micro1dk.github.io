# Encoder-Decoder Model

### 전통적 기계번역

구문 단위로 분리하여 번역한다.

```
나는 사과를 먹었습니다.
```

형태소 분석

```
나/대명사 는/조사
사과/명사 를/조사
먹다/동사 었/선어 습니다/종결어미
```

구문단위로 분리하여 번역하기 때문에 도메인 지식이 필요함.



### Encoder  - decoder model

문장 전체를 읽고 의미를 이해한 다음 번역, 인간의 방식과 유사

문장 -> 인코딩 -> Thought Vector & context vector (문맥 벡터) -> 디코딩 -> 불어..

문맥벡터는 대량의 데이터만 있으면 만들 수 있다. 따라서 도메인 전문가가 필요없다.





첨단모델은 LSTM이아닌 매우 복잡한.

최점단 모델에서도 마찬가지로 ENcoder decoder 기법을 사용함





## Decoding Strategy

* Greedy
  * softmax 분포 중 가장 높은 확률(argmax)을 선택, 언제나 같은 번역
* Sampling
  * 
* Beam-search
  * 후보군유지

고급 번역기에서는 Beam-search전략을 사용함.



한국어가 다른 어넝 대비 Corpus가 적어 품질이 떨어짐

