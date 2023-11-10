# HuggingFace Transformers library

허깅페이스는 스타트업 회사다.

구글의 Transformer를 랭귀지 모델을 기반으로한 편리하게 사용할 수 있도록 라이브러리를 제공하는 회사

NLP 기술을 많은 이들이 편히 이용할 수 있도록 기술민주화를 하는 그룹

똑같은 BERT모델을 더 편하게 FINE - TUNING을 할 수 있음



NLP와 관련된 다양한 패키지를 제공한다.

transformer에 기반한 첨단 랭귀지모델, 데이터셋, 토크나이저들. 



Transformers

* masked language model 알고리즘 제공
* pretrained 모델 배포



Tokenizers

* transformers package 에서 사용할 수 있는 subword 토크나이저 학습
* transformer와 분리되어 다른 목적에도 이용할 수 있다.



dataset -> tokenizer -> models의 언어 모델 학습에 필요한 전 과정을 지원



실습

pipeline을 이용한 downstream task 실행

* 감성분석
* 텍스트 생성
* 이름 개체 인식
* 질문 답변
* 마스킹된 텍스트 채우기
* 요약
* 번역
* 특징 추출

