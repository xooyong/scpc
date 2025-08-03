import numpy as np
import pandas as pd

import torch
from tqdm import tqdm
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration
import torchvision.transforms as transforms
from utils import ImageAugmentations, init_blip_caption_log, save_blip_caption_log, combine_caption
from config import TEST_CSV_PATH, BLIP_IMAGE_CAPTION, CFG

# 모델 및 토크나이저 로드
model_name = "google/flan-t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).eval().to("cuda")

total_params = sum(p.numel() for p in model.parameters())
print(f"총 파라미터 수: {total_params:,}")
print(f"(단위: {total_params / 1e9:.2f}B)")

# 증강 객체 생성
augmentations = ImageAugmentations()

# 적용할 증강 기법들
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

# 로그 파일 초기화
init_blip_caption_log()

print("🔄 BLIP IMAGE CAPTIONING 모델 로딩 중...")
fine_tuned_path = CFG['FINAL_MODEL']
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained(fine_tuned_path).to("cuda")
print("✅ 모델 로딩 완료!")

csv_path = TEST_CSV_PATH
df = pd.read_csv(csv_path)

results = []
debug_log = []

for index, row in tqdm(df.iterrows(), total=len(df), desc="blip image caption 생성 중...", ncols=100, colour='green'):
    img_path = row['img_path']
    img_id = row['ID']

    try:
        original_image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"⚠️ 이미지 로드 실패 (ID: {img_id}): {e}")
        results.append({"ID": img_id, "Combined_Caption": "Image load failed"})
        continue

    resized_image = transforms.Resize((384, 384))(original_image)
    captions_dict = {}
    caption_log = f"\n=== 이미지 처리 (ID: {img_id}) ===\n"
    caption_log += "각 증강 이미지에 대한 캡션:\n"

    for aug_name, aug_method in tqdm(augmentation_methods, desc=f"📸 증강처리 (ID:{img_id})", leave=False, ncols=80, colour='blue'):
        try:
            augmented_image = aug_method(resized_image)

            with torch.no_grad():
                inputs = processor(images=augmented_image, return_tensors="pt").to("cuda")
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    penalty_alpha=0.6,
                    top_k=4,
                )
                caption = processor.decode(outputs[0], skip_special_tokens=True)

            captions_dict[aug_name] = caption
            caption_log += f"  {aug_name}: {caption}\n"

            del inputs, outputs, augmented_image
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"⚠️ {aug_name} 캡션 생성 실패 (ID: {img_id}): {e}")
            captions_dict[aug_name] = "Caption generation failed"

    combined_caption = combine_caption(captions_dict, debug=False)
    caption_log += f"\n🎯 최종 결합된 캡션: {combined_caption}\n"
    caption_log += "=" * 50 + "\n"

    save_blip_caption_log(caption_log)
    debug_log.append(caption_log)

    results.append({
        "ID": img_id,
        "Combined_Caption": combined_caption
    })

    del original_image, resized_image, captions_dict
    torch.cuda.empty_cache()

# 결과 저장
output_df = pd.DataFrame(results)
output_df.to_csv(BLIP_IMAGE_CAPTION, index=False)

print(f"\n✅ 결과가 './outputs/stanford_epoch5_augmentation_captions.csv'에 저장되었습니다.")
print(f"📊 총 {len(results)}개의 이미지가 처리되었습니다.")
print(f"🎯 총 {len(results) * len(augmentation_methods)}개의 캡션을 생성했습니다.")

