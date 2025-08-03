import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.data import Dataset

from transformers import BlipProcessor, BlipForConditionalGeneration
from peft import LoraConfig, get_peft_model
torch.set_float32_matmul_precision('high')

from config import CFG, TEST_CSV_PATH

# TF32 허용
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device", device)

# LoRA 설정
lora_config = LoraConfig(
    r=CFG['LORA_R'],  # 표현력을 위해 조금 높게
    lora_alpha=CFG['LORA_ALPHA'],
    lora_dropout=CFG['LORA_DROPOUT'],
    target_modules=CFG['TARGET_MODULES']
)

processor = BlipProcessor.from_pretrained(CFG['MODEL_NAME'])

model = BlipForConditionalGeneration.from_pretrained(
    CFG['MODEL_NAME'],
    device_map=device)

total_params = sum(p.numel() for p in model.parameters())
print(f"총 파라미터 수: {total_params:,}")
print(f"(단위: {total_params / 1e9:.2f}B)")

model = get_peft_model(model, lora_config)

class ImageParagraphDataset(Dataset):
    def __init__(self, dataframe=None, csv_path=None, image_path=None, transform=None):
        if dataframe is not None:
            self.data = dataframe.reset_index(drop=True)
        elif csv_path is not None:
            self.data = pd.read_csv(csv_path, dtype={"Image_name": str})  # 여기서 문자열 처리
        else:
            raise ValueError("Either `dataframe` or `csv_path` must be provided.")

        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_name = str(row["Image_name"]) + ".jpg"  # 확장자 붙이기
        paragraph = row["Paragraph"]

        image_path = os.path.join(self.image_path, image_name)
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "paragraph": paragraph,
            "image_path": image_path
        }

def create_blip_collate_fn(processor, max_length=100):
    """
    모든 캡션을 사용하는 collate function (padding으로 길이 처리)
    """
    def collate_fn(batch):
        images = [item['image'] for item in batch]
        captions = [item['paragraph'] for item in batch]
        image_paths = [item['image_path'] for item in batch]

        encoding = processor(
            images=images,
            text=captions,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        encoding["labels"] = encoding["input_ids"].clone()
        labels = encoding["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100 
        encoding["labels"] = labels
        encoding['image_path'] = image_paths
        return encoding

    return collate_fn


df = pd.read_csv(CFG["CSV_PATH"])
train_df, valid_df = train_test_split(df, test_size=0.1, random_state=42)

train_dataset = ImageParagraphDataset(dataframe=train_df, csv_path=CFG['CSV_PATH'], image_path=CFG['IMAGE_PATH'], transform=None)
valid_dataset = ImageParagraphDataset(dataframe=valid_df, csv_path=CFG['CSV_PATH'], image_path=CFG['IMAGE_PATH'], transform=None)

collate_fn = create_blip_collate_fn(processor, max_length=100)

# ---------- 8. DataLoader 생성 ----------
train_loader = DataLoader(
    train_dataset,
    batch_size=CFG['BATCH_SIZE'],
    shuffle=True,
    collate_fn=collate_fn
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=CFG['BATCH_SIZE'],
    shuffle=False,
    collate_fn=collate_fn
)

optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['LEARNING_RATE'])
save_dir = CFG['SAVE_DIR']

for epoch in range(CFG['EPOCHS']):
    # model.trian()
    train_loss = 0.0

    # train
    for batch in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{CFG['EPOCHS']}] Training"):
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = model(
            pixel_values=batch["pixel_values"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # valid
    model.eval()
    val_loss = 0.0
    sample_results = []
    max_samples = 1  # 저장할 샘플 개수

    with torch.no_grad():
        for batch in tqdm(valid_loader, desc=f"[Epoch {epoch+1}/{CFG['EPOCHS']}] Validation"):
            sample_count = 0
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss
            val_loss += loss.item()

            # ✅ 배치당 하나씩만 생성해서 저장
            if sample_count < max_samples:
                pixel_values = batch["pixel_values"][0].unsqueeze(0)  # 1개만 추출
                label_ids = batch["labels"][0].unsqueeze(0)
                clean_label_ids = label_ids[label_ids != -100]
                image_path = batch["image_path"][0]  # 

                generated_ids = model.generate(
                    pixel_values=pixel_values,
                    max_length=CFG['MAX_LENGTH'],
                    decoder_start_token_id=processor.tokenizer.cls_token_id
                )
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                reference_text = processor.tokenizer.decode(clean_label_ids, skip_special_tokens=True)

                sample_results.append({
                    "image_path": image_path,
                    "reference_caption": reference_text,
                    "generated_caption": generated_text
                })
                if sample_count <= max_samples:
                    sample_count += 1

    avg_val_loss = val_loss / len(valid_loader)
    print(f"Train Loss: {train_loss}")
    print(f"Avg Train Loss: {avg_train_loss:.4f}")
    print(f"Valid Loss: {val_loss}")
    print(f"Val Loss: {avg_val_loss:.4f}")

    # ✅ 캡션 결과 CSV 저장
    caption_csv_path = os.path.join(save_dir, f"epoch{epoch+1}_samples.csv")
    pd.DataFrame(sample_results).to_csv(caption_csv_path, index=False, encoding="utf-8-sig")
    print(f"📄 생성 결과 저장됨: {caption_csv_path}")

    # N 에폭마다 모델 저장
    if (epoch + 1) % 1 == 0:
        save_path = os.path.join(save_dir, f"epoch{epoch+1}")
        os.makedirs(save_path, exist_ok=True)

        model.save_pretrained(save_path)
        processor.save_pretrained(save_path)
        print(f"✅ 모델 저장됨: {save_path}")

save_path = os.path.join(save_dir, f"epoch{epoch+1}")
os.makedirs(save_path, exist_ok=True)

model.save_pretrained(save_path)
processor.save_pretrained(save_path)
print(f"✅ 모델 저장됨: {save_path}")