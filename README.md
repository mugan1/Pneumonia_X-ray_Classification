### Project Title

X-Ray 폐렴 병변 분류 프로젝트 

### Overview

- 기간  |  2021. 08 ~ 2021. 09
- 팀원  |  스터디 그룹 Medi States
- 담당 파트 |  ChexNet 구현 
- 플랫폼 |  Python, Pytorch, Colab notebook

### Background 

1. 폐렴은 국내 사망 원인 3위의 질환이지만 조기 진단과 예방으로 치료가 가능한 질환임
2. 흉부 X-ray는 폐렴을 진단하는 가장 좋은 방법이며, 폐렴을 진단할 수 있는 딥러닝 모델을 통해 의료진의 진단 정확도 및 판독 시간을 단축시키기 위한 노력을 하고 있음
3. 폐렴 X-ray Dataset을 기반으로 다양한 모델링을 통해 높은 성능과 빠른 학습이 가능한 모델을 모색하고자 함

### Goal

1. 폐렴을 포함 총 14종의 병변을 진단할 수 있는 ChesXNet 모델을 활용하여 이미 검증된 모델의 성능을 확인하고, XAI를 통해 설명력을 확보하고자 함
2. Transformer를 활용한 Vision Transformer(ViT)와 MLP 개념을 활용한 MLP-Mixer가 ChesXNet의 성능과 학습 속도를 능가할 수 있을지 확인하고자 함

### Dataset

Kaggle Competition [Link](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)에서 제공한 환자별 X-ray 이미지, 폐렴 여부, 병변 부위가 좌표값으로 부여되어 있음

### Theories

1.  CheXNet 모델의 검증 결과대로 폐렴 여부를 진단하는 F1-Score는 약 70%를 달성할 것이다
2. Vit와 MLP-Mixer 모델만으로도 CheXNet 모델만큼의 성능을 달성할 수 있을 것이다

### EDA

<p align="center">
  <img src="https://github.com/mugan1/music_transcription/assets/71809159/a70965e1-d160-41e9-9a02-58bd00aad7c1)" alt="text" width="number" />
</p>

- 26684개의 이미지로 이루어진 데이터셋에서 정상(0)의 비율이 폐렴 환자(1)의 비율보다 약 2배이상 많으므로, Class가 불균형함을 알 수 있음. 따라서 평가 지표는 F1-Score로 평가 예정
  
<p align="center">
  <img src="https://github.com/mugan1/music_transcription/assets/71809159/e2859bce-b527-4851-8c9e-31d45c3d0fa5" alt="text" width="number" />
</p>

- 데이터는 dicom 파일로 제공되며 모델과의 호환을 위해 JPG 파일로 변환 작업을 거침

<p align="center">
  <img src="https://github.com/mugan1/music_transcription/assets/71809159/e5ee816c-dc1c-45e0-946a-23245f8375e8" alt="text" width="number" />
</p>

- 데이터셋에는 폐렴으로 진단된 병변 부위가 좌표값으로 주어져 있으므로, Bounding Box를 통해 병변 부위를 표시 가능
   
### CheXNet

1. 121개의 층으로 구성된 2D CNN 구조
2. 224 * 224 이미지를 벡터화
3. Class Activation Mapping (GRAD-CAM) heatmap을 활용 출력 결과 확인 가능
   - GRAD-CAM(Gradient-weighted Class Activation Mapping)은 CNN에서 분류를 진행하였을 때 분류에 가장 영향을 많이 주었던 부분을 보여주는 XAI(eXplainable Artificial Intelligence)의 일종임
4. FC를 sigmoid를 적용하여 단일 출력
5. Adam optimizer와 16개의 미니 배치, 0.001 lr 적용
6. 14개의 병변을 감지할 수 있는 모델

<p align="center">
  <img src="https://github.com/mugan1/music_transcription/assets/71809159/172eeed7-cc7c-41e2-ac54-6054202f45bf" alt="text" width="number" />
</p>

### Classification 결과 

<p align="center" width="100%">
  <img src="https://github.com/mugan1/music_transcription/assets/71809159/caa05a6c-821b-48fc-8086-1c2611cd379e" alt="text" width="number" width="40%"/>
  <img src="https://github.com/mugan1/music_transcription/assets/71809159/40b78226-135d-4647-9577-fa52fb917da7" alt="text" width="number" width="20%"/>
</p>

1. Accuracy 0.83, F1- Score 0.70
2. Confusion Matrix를 보면 알 수 있듯이, 상대적으로 Ground Truth가 폐렴인 경우 잘못 예측한 비율(0.62)이 제대로 예측한 비율(0.38)보다 높게 나타남(False negative)

### Feature Selection

사용자 입력화면에서 받을 데이터 수를 줄여야할 필요성이 있으므로 sklearn의 Select K-Best을 사용하여 주요 변수를 11개만 추출
- Select K-Best : Feature Selection의 일종으로 Target 변수와의 상관관계를 계산하여 가장 중요하다고 판단되는 변수를 K개 산출하는 방식
- 선택된 최종 변수 : 연식, 주행거리, 연료, 배기량, 마력, 최대토크, 제조사, 보증여부, 보험이력등록, 구동방식, 연비
  
### CheXNet Visualization

1. Cherry Picked Examples
   
<p align="center" width="100%">
  <img src="https://github.com/mugan1/music_transcription/assets/71809159/f9b3d127-9b3e-4ea6-bc70-b8f18bffc661" alt="text" width="number" width="30%"/>
  <img src="https://github.com/mugan1/music_transcription/assets/71809159/6b532550-7efb-45b2-9e9e-5dc46785edfe" alt="text" width="number" width="30%"/>
</p>

<p align="center" width="100%">
  <img src="https://github.com/mugan1/music_transcription/assets/71809159/2753baec-5718-43f4-a791-db830faffe54" alt="text" width="number" width="30%"/>
  <img src="https://github.com/mugan1/music_transcription/assets/71809159/b98e4a3d-b400-4d92-83c9-9c600892bb18" alt="text" width="number" width="30%"/>
</p>

- GRAD-CAM으로 시각화 한 결과, 실제 병변 부위 모두를 정확히 인식하지는 못했지만, 폐렴으로 진단할 경우 병변 부위를 붉은 색으로 집중적으로 표시함으로써 설명력을 제공함

2. Lemon Picked Examples

<p align="center" width="100%">
  <img src="https://github.com/mugan1/music_transcription/assets/71809159/a8458799-4bc8-4d38-aab2-bbc3049e839a" alt="text" width="number" width="30%"/>
  <img src="https://github.com/mugan1/music_transcription/assets/71809159/3fd3f355-b7e9-47b8-92ae-d21bce0749fd" alt="text" width="number" width="30%"/>
</p>

<p align="center" width="100%">
  <img src="https://github.com/mugan1/music_transcription/assets/71809159/02e4bd54-438d-4a10-abf3-de09454cfd8a" alt="text" width="number" width="30%"/>
  <img src="https://github.com/mugan1/music_transcription/assets/71809159/4402b9bb-3f1f-4311-bfb9-6140a91f9dc3" alt="text" width="number" width="30%"/>
</p>

- 잘못 예측한 경우는 대부분 False negative였으며, 정상으로 오인하였을 시 모델은 병변 부위를 제대로 인식하지 못해 붉은 색의 표시가 산발적으로 나타남
  
### Vision Transformer

<p align="center">
  <img src="https://github.com/mugan1/music_transcription/assets/71809159/a40afce6-498e-4b88-9545-af190d030e0c" alt="text" width="number" />
</p>

1. ViT의 핵심은 Image Patches를 문장의 Tokens처럼 Embedding하여 Transformer 모델에 적용시킬 수 있다는 것임
2. Transformer의 Encoder 부분만을 사용하며 MLP를 통해 Classification 진행
3. SoTA보다 뛰어난 성능을 약 15분의 1의 계산 비용으로 달성할 수 있다는 큰 장점이 있음
4. 단, 사전학습량과 모델이 충분히 클 경우에만 성능 개선이 가능함

### MLP-Mixer 

<p align="center">
  <img src="https://github.com/mugan1/music_transcription/assets/71809159/fae8c816-44f1-4218-9a1d-3f6fd3ad296d" alt="text" width="number" />
</p>

1. ViT의 Attention도 필요없이 MLP 개념만으로도 충분한 성능을 달성할 수 있다는 개념에서 나온 모델
2. MLP-Mixer는 Token-mixing MLP와 Channel-mixing MLP, 2가지 Layer가 존재함
3. Token-mixing MLP block 은 행 특성끼리 공유되며, Channel-mixing MLP block 은 열 특성끼리 공유됨. 
4. 별도의 Positional Encoding이 필요하지 않는 이유는, Token mixing 을 할 때 이미지 패치별 특성 위치 그대로 layer 계층에 입력되기 때문임
5. Inductive Bias가 줄기 때문에 제약으로부터 자유로운 모델

### Inductive Bias

1. 귀납적 편향으로써 학습자가 경험하지 않은 주어진 입력의 출력을 예측하는 데 사용하는 가정의 집합
2. 분산이 크면 훈련 데이터에 지나치게 적합을 시킨 과적합이 발생하며, 편향이 클 경우 과소적합이 발생함. 이 둘은 Trade-off 관계임
3. CNN의 구조나 ViT의 Attention은 학습과정에서 추가적인 가정을 하게 되는 요인이며(Inductive Bias) 이를 줄이는 연구 결과 중 하나가 MLP-Mixer임

<p align="center">
  <img src="https://github.com/mugan1/music_transcription/assets/71809159/39434cb1-db7b-413f-b97f-0fcd6ceaf048" alt="text" width="number" />
</p>

### Classification 결과 비교

<p align="center">
  <img src="https://github.com/mugan1/music_transcription/assets/71809159/8fb95e5e-587e-41b5-b577-3fa12e3e1f44" alt="text" width="number" />
</p>

1. 세 모델 모두 Pre-trained Model을 Fine-Tuning한 결과
2. MLP-Mixer가 F1-Score가 0.72로 가장 높은 성능을 보임

### Conclusion

1. 총평
  - ChexNet의 F1-Score는 약 0.70으로 가설을 입증하였으며, GRAD-CAM을 통해 설명 가능한 AI를 구현할 수 있었음
  - ViT와 MLP-Mixer 역시 Classifaction에 있어서는 121개의 CNN으로 구성된 ChexNet과 비교하여 비슷하거나 뛰어난 성능을 보임
2. 소감 및 기대효과

  - 세 모델의 F1-Score 점수가 0.70 안팎에 머문 것은 폐렴 진단 Dataset이 부족하여 병변의 특징추출을 충분히 학습하지 못한 것이라 판단되며, 충분한 데이터셋의 확보는 곧 성능향상으로 이어질 것으로 예상함
  - 많은 양의 사전학습을 전제로 ViT와 MLP-Mixer도 CNN보다 빠르고 정확한 Classification이 가능함을 엿볼 수 있었음
  - GRAM-CAM과 같은 XAI는 설명력이 요구되는 의료 AI 분야에서 많은 수요와 발전이 있을 것으로 예상됨

### References

1. CheXNet  : [Link](https://arxiv.org/abs/1711.05225) 
2. Vision Transformer : [Link](https://arxiv.org/abs/2010.11929)
3. MLP-Mixer : [Link](https://arxiv.org/abs/2105.01601)
   
