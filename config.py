# test data 경로
TEST_CSV_PATH = "./data/test.csv"
TEST_IMAGE_DIR = "./data/test_input_images"

# 추론 결과 저장 경로
SUBMISSION = "./outputs/submission.csv"

# 추론 결과 로그 저장 경로
SUBMISSION_LOG = "./logs/t5_answers.txt"

# blip image captioning, vqa 모델을 통해서 생성된 결과물 저장 경로
BLIP_IMAGE_CAPTION = "./outputs/stanford_epoch5_augmentation_captions.csv"
BLIP_QA_PAIRS = "./outputs/blip_vqa_augmentation_results.csv"

# blip image captioning, vqa 모델 로그 저장 경로
BLIP_IMAGE_CAPTION_LOG = "./logs/caption_debug_log.txt"
BLIP_VQA_LOG = "./logs/vqa_debug_log.txt"

# blip image captioning model fine tuning hyperparameters
CFG = {
    'MODEL_NAME': "Salesforce/blip-image-captioning-large",
    "CSV_PATH": './data/archive/stanford_df_rectified.csv',
    'IMAGE_PATH': './data/archive/stanford_img/content/stanford_images/',
    'SAVE_DIR': './models/checkpoints', # 모델 저장 위치
    'FINAL_MODEL': './models/checkpoints/epoch5', # 최종적으로 사용할 모델
    'BATCH_SIZE': 12,
    'EPOCHS': 10,
    'LEARNING_RATE': 2e-4,
    'MAX_LENGTH': 100,
    'LORA_R': 16,
    'LORA_ALPHA': 32,
    'LORA_DROPOUT': 0.1,
    'TARGET_MODULES': ['query', 'key', 'value', 'qkv'],
}