## BLEU (Bilingual Evaluation Understudy)

기계번역의 품질을 평가하는 알고리즘. 언어에 무관하며 이해하기 쉽고 계산하기 쉽다.

0~1 값

### Calculating Score

unigram or bigram

* references: "the weather is extremely good"

  (the, weather), (weather, is) ...

  2단어로 짝을 맞춰서

* predicted result = "the weather is good"

  (the weather), (weather, is), (is, good)

* BLEU = 1/3 + 1/3 + 0/3 = 0.666



## Teacher Forcing

교사학습. 선생이 학생 가르치는 가르치는 것

* 첫 번째 단어에서 잘못된 예측을 했을 경우 시간이 지날수록더 큰 잘못예측을 할 가능성이 증가.
* 시퀀스 구성이 바뀌면서, 예측 시퀀스 길이도 바뀜
* 학습 과정에서는 이미 정답을 알고 있고(지도학습이기에), 현재 모델의 예측값과 정답과의 차이를 통해 학습하므로, 실제 값을 다음 단어 예측의 입력 값으로 사용한다.
* 단계별 수정.
* 전 단계의 출력이 다음단계의 입력이 일반적인 방법인데, 훈련단계에서는 정답값을 다음 예측값의 입력값으로 하는 방법



I Love you의 경우 I를 먼저 예측해야함. 

Dropout과 마찬가지로 훈련에서 적용하는 기법. 모델을 훈련용, product용으로 2가지를 만들어야함





Encoder - Decoder 는 모델을 2개 만든다.



