# Attender: Model Specifications
  * <strong>Summarization</strong> with a designated token-length + <strong>Segregation</strong> of sentences.
## basics
* Bart base
  * fine-tuned pre-trained KoBart model of "gogamza/kobart-base-v1" loaded from huggingface api (https://huggingface.co/gogamza/kobart-base-v1)
  * [bart_modified](./model/bart_modified) is the modified files' directory.
* Summary model is based on the KoBart, while Segment model has its own structure, which consists of transformer encoder layers that recieve and processs KoBart encoder output.
* total trainable parameters: 134,501,378

## Segment Model
* input: receive base model(Bart)'s encoder output
* structure: positional_encoding -> transformer encoder layers(num=2) -> last_projection(last_dim=2)
* target: for each token, predict if it is a boundary(=1) or not(=0)

## Summary Model
* input: source-ids, source-masks, desired-summary-lenghts for each segment
* structure: Bart-base
  * encoder layers=6, decode layers=6
  * d_model=768, ffn_dim=3072, 
  * num_attention_heads=16
* Additional Methods:
  * Inside the [modeling_bart.py](model/bart_modified/modeling_bart.py):<br>
  1. <strong>_laam_attn_is_multi()</strong>: Implementing the LAAM method proposed in the paper of [Liu et al. 2022](https://aclanthology.org/2022.acl-long.474.pdf): Multipy length-aware weighting matrix to the attention weights matrix.<br>
      * This process works during the decoder's cross-attention mechanism.
      * lenght-aware weighting matrix is sent to the model from the outside.
  2. <strong>_attention_filter()</strong>: Multiplying weighting filters per each segments.<br>
      * Works during the decoder's cross-attention mechanism.
      * weighting filter is created outside of the decoder from a combination of the encoder's output and segment model's output. Then it is sent to the decoder.

# Training
* Here, segment model and summary model are trained seperately and combined afterward.

# Test Examples

[test_src/news_1.txt](./test/test_src/news_1.txt) 

```test_src/news_1.txt, 20, 50 ```
```
Summaries:
  20-tokens: <s> 배런스는 중국 정부가 어떤 조치를 취하더라도 중국 경제가 과거의 고성장 시대로 돌아갈 가능성은 작다고 말했다.
  50-tokens: <s> 중국 경제 전문가들은 중국 경제의 구조적인 문제 때문에 경기 둔화는 경기회복이 지연된 문제라고 분석하며 비구이위안 사태로 중국 경제는 디플레이션에 처해있고 디플레이션은 디플레이션으로 이어질 수 있다고 분석했다.</s>

Segments:
  SEG1 <s> 각종 악재에 직면한 중국 경제를 ~ 그러나 뚜껑을 열고 보니
  SEG2 중국 경제성장을 이끄는 '삼두마차'인 소비, 투자, 수출 모두에서 기대 이하의 성적표가 나오면서 ~ 대만 당국은 비구이위안 사태가 대만 경제에 미칠 파장에도 촉각을 곤두세우고 있다.
```

[test_src/news_2.txt](./test/test_src/news_2.txt)

```test_src/news_2.txt, 20, 40```
```
Summaries:
  20-tokens: <s> 기상청은 동해안뿐만 아니라 전국의 해안지역에선 긴장을 늦춰선 안된다고 주의를 당부했다.
  40-tokens: <s> 기상청은 태풍이 우리나라 남부 지역을 강타할 가능성이 높다고 보고 산사태 위기 수준을 상향했고 산사태 위기 경보 수준을 산사태 위기에서 경계로 상향 조정했으며 태풍 피해도 우려했다.</s>

Segments:
  SEG1 <s> 5일 오후∼6일 새벽 우리나라 남부 지역을 강타한 태풍 ‘힌남노’가 북동쪽으로 이동했다. ~ 약해진 태풍의 자리를 북쪽에서 내려온 차갑고 건조한 공기가 차지해 풍속이 초속 20∼40m에 달할 것으로 관측된다.
  SEG2 태풍에서 멀리 떨어져 있는 인천 역시 안전할 수 없는 이유다. ~ 고 경고했다.</s>
```

[test_src/tesla.txt](./test/test_src/tesla.txt)

```test_src/tesla.txt, 30, 40, 50  ```
```
Summaries:
  30-tokens: <s> 전기공학자이자 발명가인 그라츠 공과 대학교에서 수학한 그는 전기 아크등 조절기 특허권을 획득했고 다이너모 했다.
  40-tokens: <s> 테슬라코일은 고주파 진동 전류를 공심 변압기로 승압하여 교류의 고전압을 발생시키는 장치이며 특허는 굴리히 1900년부터 획득했다.
  50-tokens: 존스는 독보적인 발명가이며 독창적인 발명가로 알려져 있으며 그의 독창적인 아이디 어는 독보적인 물리학자로 알려져 있으며 그의 독창적인 발명품은 비행기 제작에 대한 특허를 획득했다.</s>

Segments:
  SEG1 <s> 전기공학자이자 발명가로, 1856년 오스트리아 제국의 스밀랸(현 크로아티아 영역)에서 세르비아인 부모의 5자녀 중 넷째로 태어났다. ~ 이  갈등은 전류에 대한 생각의 차이, 즉 에디슨은 직류, 테슬라는 교류전기 시스템이 우수하다고 확신한 데에서 비롯되 었다.
  SEG2 1893년 시카고에서 열린 컬럼비아세계박람회(시카고박람회)에 교류 전기가 채택되고, ~ 금융업자 존 모건의 투자를 받 아 뉴욕주 롱아일랜드 쇼어햄에 무선전신탑인 워든클리프 탑(Wardenclyffe Tower, 일명
  SEG3 테슬라 탑)을 세우는 작업을 시작하였다. 전세계로 정보와 전력을 전송하는 시스템을 구축하려 한 이 프로젝트는 ~ 자기력선속의 밀도를 나타내는 국제단위인 테슬라(tesla, 기호: T)는 그의 이름에서 비롯되었다. 그밖에 세르비아 수도 베오그라드에 있는 ‘니콜라테슬라국제공항’, 베오그라드의 과학</s>
```


