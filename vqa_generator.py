
import numpy as np
import pandas as pd

import torch
from tqdm import tqdm
from transformers import BlipProcessor, BlipForQuestionAnswering, BlipForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration
from datetime import datetime
from PIL import Image
import torchvision.transforms as transforms

from utils import ImageAugmentations, init_vqa_log, save_blip_vqa_log, combine_vqa
from config import TEST_CSV_PATH, BLIP_QA_PAIRS

# 모델 및 토크나이저 로드
model_name = "google/flan-t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).eval().to("cuda")

total_params = sum(p.numel() for p in model.parameters())
print(f"총 파라미터 수: {total_params:,}")
print(f"(단위: {total_params / 1e9:.2f}B)")

# 1. 랜덤 seed 생성
seed = 2746317213
torch.manual_seed(seed)

# 4. 5개의 질문 생성
questions = []
for i in range(6):
    types = ['What', 'What', 'Who', 'Who', 'Where', 'Where']
    prompt = f"""
        "You are given an image, but you cannot see it. "
        "To understand what the image contains, generate one concise question. "
        "that begins with {types[i]}. "
        "Only generate the question."
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(
        input_ids,
        penalty_alpha=0.6,
        top_k=4,
        do_sample=True,
        max_new_tokens=50  # 질문 길이 제한
    )
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    questions.append(question)

# 5. 출력
print(f"사용된 seed: {seed}")
print("모델 출력:")
for i, question in enumerate(questions, 1):
    print(f"생성된 질문 {i}: {question}")

# 7. 증강 객체 생성
augmentations = ImageAugmentations()

# 8. 적용할 증강 기법들
augmentation_methods = [
    ("Original", augmentations._original),
    ("Brightness_Up", augmentations._brightness_up),
    ("Contrast_Up", augmentations._contrast_up),
    ("Top-Left", augmentations._crop_top_left),
    ("Top-Right", augmentations._crop_top_right),
    ("Bottom-Left", augmentations._crop_bottom_left),
    ("Bottom-Right", augmentations._crop_bottom_right),
    ("Center Fixed", augmentations._crop_center_fixed)
]

# BLIP VQA 모델 및 프로세서 로드
print("🔄 BLIP VQA 모델 로딩 중...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to("cuda")
print("✅ 모델 로딩 완료!")

init_vqa_log()

import os
# CSV 파일 로드
csv_path = TEST_CSV_PATH
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path, header=0, names=["ID", "img_path", "Question", "A", "B", "C", "D"])

# 미리 정의된 질문들
predefined_questions = questions

# 이미지 처리 및 VQA 생성
results = []
debug_log = []

for index, row in tqdm(df.iterrows(), total=len(df), desc="blip vqa 생성 중...", ncols=100, colour='green'):
    img_path = row['img_path']
    img_id = row['ID']

    try:
        original_image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"⚠️ 이미지 로드 실패 (ID: {img_id})")
        result_row = {"ID": img_id}
        for i, question in enumerate(predefined_questions):
            result_row[f"Question_{i+1}"] = question
            result_row[f"Combined_Answer_{i+1}"] = "Image load failed"
        for col in ['Question', 'A', 'B', 'C', 'D']:
            if col in row:
                result_row[col] = row[col]
        results.append(result_row)
        continue

    resized_image = transforms.Resize((384, 384))(original_image)
    vqa_log = f"\n=== 이미지 VQA 처리 (ID: {img_id}) ===\n"
    result_row = {"ID": img_id}

    for q_idx, question in enumerate(predefined_questions):
        vqa_log += f"\n--- 질문 {q_idx+1}: {question} ---\n"
        answers_dict = {}

        for aug_name, aug_method in tqdm(augmentation_methods, desc=f"📸 Q{q_idx+1} (ID:{img_id})", leave=False, ncols=80, colour='blue'):
            try:
                augmented_image = aug_method(resized_image)
                inputs = processor(images=augmented_image, text=question, return_tensors="pt").to("cuda")
                outputs = model.generate(**inputs, max_new_tokens=30, penalty_alpha=0.6, top_k=4)
                answer = processor.decode(outputs[0], skip_special_tokens=True)
                answers_dict[aug_name] = answer
                vqa_log += f"  {aug_name}: {answer}\n"
                del inputs, outputs, augmented_image
                torch.cuda.empty_cache()
            except Exception as e:
                error_msg = f"VQA generation failed: {e}"
                answers_dict[aug_name] = error_msg
                vqa_log += f"  {aug_name}: {error_msg}\n"

        combined_answer = combine_vqa(answers_dict, debug=False)
        result_row[f"Question_{q_idx+1}"] = question
        result_row[f"Combined_Answer_{q_idx+1}"] = combined_answer
        vqa_log += f"🎯 최종 결합된 답변: {combined_answer}\n"

    vqa_log += "=" * 60 + "\n"

    # ✅ 실시간 로그 저장
    save_blip_vqa_log(vqa_log)

    debug_log.append(vqa_log)
    results.append(result_row)

    del original_image, resized_image
    torch.cuda.empty_cache()

# 🔥 처리 완료 로그 추가
completion_log = f"\n{'='*80}\n🎉 VQA 처리 완료!\n⏰ 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n📊 총 {len(results)}개 이미지 처리됨\n{'='*80}\n"
save_blip_vqa_log(completion_log)

# 최종 결과 저장
output_df = pd.DataFrame(results)
output_df.to_csv(BLIP_QA_PAIRS, index=False)

print(f"\n✅ 결과가 './outputs/blip_vqa_augmentation_results.csv'에 저장되었습니다.")
print(f"📊 총 {len(results)}개의 이미지가 처리되었습니다.")
print(f"🎯 총 {len(results) * len(predefined_questions) * len(augmentation_methods)}개의 VQA를 생성했습니다.")
print(f"❓ 각 이미지당 {len(predefined_questions)}개의 질문에 대한 답변이 생성되었습니다.")
print(f"📝 상세 로그가 './logs/vqa_debug_log.txt'에 저장되었습니다.")

print(f"\n📋 생성된 CSV 구조:")
print(f"   - ID: 이미지 식별자")
for i, question in enumerate(predefined_questions):
    print(f"   - Question_{i+1}: {question}")
    print(f"   - Combined_Answer_{i+1}: 해당 질문에 대한 결합된 답변")

