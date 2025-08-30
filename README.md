# scpc
2025 Samsung Collegiate Programming Challenge : AI 챌린지

### **[🏆 대회 솔루션 PPT](https://github.com/xooyong/scpc/blob/main/%EB%B3%B8%EC%84%A0%20%EB%B0%9C%ED%91%9C%EC%9E%90%EB%A3%8C.pdf)**
# 🔧 환경 구성 및 실행 가이드

## 📋 시스템 환경
- **OS**: Ubuntu 22.04.5 LTS (`jammy`)
- **CUDA**: 12.4 (nvcc v12.4.131)
- **Python**: 3.12+

## 📦 가상환경 설정

```bash
# Python venv 모듈 설치
sudo apt update
sudo apt install python-venv

# 가상환경 생성
python -m venv venv

# 가상환경 활성화
source venv/bin/activate

# 필요한 패키지 설치
pip install -r requirements.txt

```

# 🧠 이미지 캡셔닝 + VQA 모델 솔루션

본 프로젝트는 이미지 기반 4지선다형 VQA(Vision Question Answering) 문제를 해결하기 위해,
이미지 캡셔닝 모델, VQA 모델과 텍스트 기반 생성 모델(FLAN-T5)을 조합한 파이프라인입니다.



## 📌 주요 기능
- 이미지에서 캡션과 VQA 출력 생성 (BLIP 사용)
- 캡션 + 질문 + 선택지 → 정답을 생성하는 텍스트 모델 (google/flan-t5-large)
- 제출용 CSV 생성 기능 포함



## 🧱 사용된 AI 모델

| Task | Model | 설명 |
|------|-------|------|
| 이미지 캡셔닝 | blip-image-captioning-large | 이미지에 대한 설명 생성 |
| VQA | blip-vqa-base | 이미지에 대해서 질문에 대한 답 생성 |
| 정답 생성 | FLAN-T5 | 질문에 대해 A/B/C/D 중 하나 생성 |
| 학습 기법 | LoRA | 경량 파인튜닝 |



## 📂 디렉토리 구조

```bash
scpc/
├── data/ # 입력 이미지 및 CSV(test/train)
├── models/ # 학습된 blip-image-captioning-large 모델 가중치 (.safetensors)
├── outputs/ # 예측 결과 저장
├── .gitignore
├── README.md
├── caption_generator.py # 이미지 캡션 생성
├── config.py # 프로젝트 전체에서 사용하는 경로, 모델명, 하이퍼파라미터 설정
├── inference.py # 추론
├── requirements.txt # 필요한 패키지 목록
├── train.py # 이미지 캡셔닝 모델(BLIP) 학습
├── utils.py # 로그 생성, 이미지 증강 등의 공통 함수 정의
└── vqa_generator.py # VQA 출력 생성
```

> `data/` 디렉토리 내부에는 테스트 이미지가 저장된 `test_input_images/`디렉토리와 `test.csv` 파일이 존재해야 합니다.

## 🏋️ 학습
```bash
python train.py
```
> `data/` 디렉토리 내부에는 학습 데이터가 저장된 `archive/`가 존재해야 되고 이 내부에는 `stanford_img`와 `stanford_df_rectified.csv`가 존재해야 합니다.

### 🔗 학습 데이터 다운로드
- [stanford Image Paragraph Captioning dataset](https://www.kaggle.com/datasets/vakadanaveen/stanford-image-paragraph-captioning-dataset)


## 🚀 실행
### 1. 이미지 캡션 생성(필요 시)
```bash
python caption_generator.py
```
### 2. VQA 출력 생성(필요 시)
```bash
python vqa_generator.py
```
> outputs/에 이미지 캡션(stanford_epoch5_augmentation_captions.csv)과 VQA 출력(blip_vqa_augmentation_results.csv)이 미리 저장되어 있으므로 필요 시에 이미지 캡션과 VQA 출력 생성
### 3. 추론
```bash
python inference.py
```
