
import pandas as pd
import numpy as np
import os
import torch
import random
import re
from difflib import SequenceMatcher
from collections import Counter
from config import TEST_CSV_PATH, BLIP_IMAGE_CAPTION, BLIP_QA_PAIRS, SUBMISSION, SUBMISSION_LOG
from datetime import datetime
from tqdm import tqdm

from transformers import T5Tokenizer, T5ForConditionalGeneration

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42) # Seed 고정

def make_prompt_combined_with_hints(caption1, question, options, qa_pairs=None, use_blip_caption=True, use_qa_pairs=True):
    """캡션, 질문, 옵션을 결합하여 프롬프트 생성"""

    # 프롬프트 시작부
    prompt = """""" # You are an AI assistant that analyzes image captions and answers multiple-choice questions.\n\n

    # BLIP caption을 질문 다음에 추가
    if use_blip_caption and caption1 and caption1 != "No caption available":
        prompt += f"Image Caption: {caption1.strip()}\n"

    # QA 쌍들 추가
    if use_qa_pairs and qa_pairs:
        info_counter = 1
        for i in range(1, 7):  # Question_1부터 Question_6까지
            question_key = f"Question_{i}"
            answer_key = f"Combined_Answer_{i}"

            if question_key in qa_pairs and answer_key in qa_pairs:
                q = qa_pairs[question_key]
                a = qa_pairs[answer_key]
                if q and str(q).strip() and a and str(a).strip():
                    prompt += f"Additional Hint: {str(q).strip()} ({str(a).strip()})\n" # Image Caption {info_counter + 1}
                    info_counter += 1
        prompt += "\n"

    # 질문을 먼저 추가
    prompt += f"Q: {question.strip()}\n"

    # 옵션들을 마지막에 추가
    opt_str = "\n".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(options)])
    prompt += f"Options:\n{opt_str}\n\n" # Options:\n

    # 정답 유도
    prompt += "A: Let's think step by step."

    return prompt

def generate_prompts_from_csv(
    test_csv_path,
    blip_caption_path,
    qa_pairs_path,
    use_blip_caption,
    use_qa_pairs
):
    """CSV 파일들에서 데이터를 읽어 프롬프트 생성"""

    print(f"📁 파일 경로 확인:")
    print(f"  - test_csv_path: {test_csv_path} (존재: {os.path.exists(test_csv_path)})")
    print(f"  - blip_caption_path: {blip_caption_path} (존재: {os.path.exists(blip_caption_path)})")
    print(f"  - qa_pairs_path: {qa_pairs_path} (존재: {os.path.exists(qa_pairs_path)})")
    print()

    # CSV 파일 로드 및 유효성 확인
    qa_df = None
    blip_caption_df = None
    qa_pairs_df = None

    if os.path.exists(test_csv_path):
        # 수정된 컬럼 순서: ID, img_path, Question, A, B, C, D
        qa_df = pd.read_csv(test_csv_path, header=0, names=["ID", "img_path", "Question", "A", "B", "C", "D"])
        print(f"✅ test CSV 로드 완료: {len(qa_df)}개 행")
        print(f"   첫 번째 행: {qa_df.iloc[0].to_dict()}")
    else:
        print(f"❌ test_csv_path가 존재하지 않습니다: {test_csv_path}")
        return []

    if use_blip_caption and os.path.exists(blip_caption_path):
        blip_caption_df = pd.read_csv(blip_caption_path)
        print(f"✅ BLIP caption CSV 로드 완료: {len(blip_caption_df)}개 행")
    elif use_blip_caption:
        print(f"⚠️ blip_caption_path가 존재하지 않습니다: {blip_caption_path}")

    # QA pairs CSV 로드
    if use_qa_pairs and qa_pairs_path and os.path.exists(qa_pairs_path):
        qa_pairs_df = pd.read_csv(qa_pairs_path)
        print(f"✅ QA pairs CSV 로드 완료: {len(qa_pairs_df)}개 행")
    elif use_qa_pairs:
        print(f"⚠️ qa_pairs_path가 존재하지 않습니다: {qa_pairs_path}")

    # Dict 변환
    blip_caption_dict = dict(zip(blip_caption_df["ID"], blip_caption_df["Combined_Caption"])) if blip_caption_df is not None else {}

    # QA pairs dict 변환
    qa_pairs_dict = {}
    if use_qa_pairs and qa_pairs_df is not None:
        for _, row in qa_pairs_df.iterrows():
            qa_pairs_dict[row["ID"]] = row.to_dict()

    # 결과 저장
    prompts = []

    if qa_df is None:
        print("❌ qa_csv_path가 유효하지 않아 프롬프트 생성을 중단합니다.")
        return prompts

    print(f"\n🔄 프롬프트 생성 중...")
    for _, row in qa_df.iterrows():
        image_id = row["ID"]

        # 캡션
        caption1 = blip_caption_dict.get(image_id, "No caption available") if use_blip_caption else ""

        # 질문
        question = row["Question"]

        # 옵션
        options = [row["A"], row["B"], row["C"], row["D"]]

        # QA pairs 가져오기
        qa_pairs = qa_pairs_dict.get(image_id, {}) if use_qa_pairs else {}

        caption1 = caption1 if caption1 != "No caption available" else ""

        prompt = make_prompt_combined_with_hints(
            caption1, question, options, qa_pairs,
            use_blip_caption=use_blip_caption,
            use_qa_pairs=use_qa_pairs
        )
        prompts.append({"ID": image_id, "prompt": prompt, "question": question})

    print(f"✅ 총 {len(prompts)}개 프롬프트 생성 완료\n")
    return prompts

# 실행 부분
if __name__ == "__main__":
    # 프롬프트 생성
    prompts = generate_prompts_from_csv(
        test_csv_path=TEST_CSV_PATH,
        blip_caption_path=BLIP_IMAGE_CAPTION,
        qa_pairs_path=BLIP_QA_PAIRS,
        use_blip_caption=True,
        use_qa_pairs=True
    )

    if prompts:
        print(f"📋 처음 5개 프롬프트 예시:\n")
        print("=" * 80)

        # 프롬프트 출력
        for i, prompt in enumerate(prompts[:10]):
            print(f"[{i+1}] ID: {prompt['ID']}")
            print(f"Prompt:\n{prompt['prompt']}")
            print("-" * 80)
    else:
        print("❌ 생성된 프롬프트가 없습니다.")


# Inference
print("🚀 T5 모델 로딩 시작...")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large").to("cuda")
print(f"✅ T5 모델 로딩 완료! 총 {len(prompts)}개 프롬프트 처리 시작...")

# 결과 저장 리스트
results = []

# 답변 로그 파일 초기화
answer_log_filename = SUBMISSION_LOG
with open(answer_log_filename, "w", encoding="utf-8") as answer_log:
    answer_log.write("=== T5 모델 답변 로그 ===\n")
    answer_log.write(f"총 {len(prompts)}개 프롬프트 처리\n")
    answer_log.write(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    answer_log.write("=" * 50 + "\n\n")

# 추론 루프
for i, item in enumerate(tqdm(prompts, desc="T5 추론 중"), 1):
    prompt = item["prompt"]
    id_ = item["ID"]
    question = item["question"]

    # 토크나이징
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

    # 생성
    outputs = model.generate(input_ids, max_new_tokens=256, do_sample=False)

    # 디코딩
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = generated_text.strip()

    # 콘솔 출력
    # print(f"[{i}/{len(prompts)}] {id_}: '{answer}'")

    # 답변 로그 파일에 실시간 저장
    with open(answer_log_filename, "a", encoding="utf-8") as answer_log:
        answer_log.write(f"[{i:03d}] ID: {id_}\n")
        answer_log.write(f"Question: {question}\n")
        answer_log.write(f"Answer: {answer}\n")
        answer_log.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        answer_log.write("-" * 40 + "\n\n")

    results.append({"ID": id_, "answer": answer})

# 답변 로그 파일 마무리
with open(answer_log_filename, "a", encoding="utf-8") as answer_log:
    answer_log.write("=" * 50 + "\n")
    answer_log.write(f"처리 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    answer_log.write(f"총 처리된 답변 수: {len(results)}\n")

# CSV 저장
output_filename = SUBMISSION
submission_df = pd.DataFrame(results)
submission_df.to_csv(output_filename, index=False, encoding='utf-8')
print(f"\n✅ 총 {len(results)}개 결과가 {output_filename}에 저장되었습니다.")
print(f"📝 답변 로그가 {answer_log_filename}에 저장되었습니다.")

# 답변 추출
def similarity(a, b):
    """두 문자열의 유사도를 계산 (0~1)"""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

def extract_keywords(text, min_length=3):
    """텍스트에서 키워드 추출 (불용어 제거)"""
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'a', 'an', 'this', 'that', 'these', 'those', 'it', 'its', 'you', 'your', 'i', 'my', 'me', 'we', 'our', 'they', 'their', 'them', 'he', 'his', 'him', 'she', 'her', 'as', 'if', 'when', 'where', 'why', 'how', 'what', 'who', 'which', 'than', 'so', 'very', 'just', 'now', 'then', 'here', 'there'}

    # 문장부호 제거 및 단어 분할
    clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
    words = [w.strip() for w in clean_text.split() if len(w) >= min_length and w not in stop_words]
    return words

def advanced_extract_answer(text, options, question=""):
    """
    고급 정답 추출 함수
    """

    # 1단계: 확장된 전통적 패턴들
    traditional_patterns = [
        # 기존 패턴들
        r"(?:the\s+)?answer\s+is\s+\(([A-D])\)",                    # The answer is (C)
        r"(?:so[,]?\s+)?(?:the\s+)?answer\s+is\s+\(([A-D])\)",      # So the answer is (C)
        r"(?:the\s+)?answer[:\s]*\(([A-D])\)",                       # The answer: (C)
        r"(?:so[,]?\s+)?(?:the\s+)?final\s+answer\s+is\s+\(([A-D])\)", # So the final answer is (C)
        r"answer[:\s]*\(([A-D])\)",                                  # Answer: (A)
        r"^\(([A-D])\)",                                             # (A) 시작
    ]
    text_lower = text.lower().strip()

    for pattern in traditional_patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # 2단계: 정확한 텍스트 매치 (개선된 버전)
    exact_matches = []
    for i, option in enumerate(options):
        if option and str(option).strip():
            option_clean = str(option).strip().strip('"').strip("'")

            # 다양한 형태로 텍스트 정규화
            option_normalized = re.sub(r'[^\w\s]', ' ', option_clean.lower())
            text_normalized = re.sub(r'[^\w\s]', ' ', text_lower)

            # 정확한 매치
            if option_clean.lower() in text_lower:
                exact_matches.append((chr(65+i), len(option_clean), 1.0))
            # 정규화된 매치
            elif option_normalized in text_normalized:
                exact_matches.append((chr(65+i), len(option_clean), 0.9))

    if exact_matches:
        exact_matches.sort(key=lambda x: (x[2], x[1]), reverse=True)
        return exact_matches[0][0]

    # 3단계: 향상된 유사도 기반 매치
    similarity_matches = []
    for i, option in enumerate(options):
        if option and str(option).strip():
            option_clean = str(option).strip().strip('"').strip("'")

            # 전체 유사도
            sim_full = similarity(text, option_clean)

            # 핵심 부분 유사도 (텍스트의 마지막 부분과 비교)
            text_words = text.split()
            if len(text_words) > 10:
                text_last_part = ' '.join(text_words[-len(option_clean.split()):])
                sim_partial = similarity(text_last_part, option_clean)
            else:
                sim_partial = sim_full

            max_sim = max(sim_full, sim_partial)

            if max_sim >= 0.2:  # 임계값 낮춤
                similarity_matches.append((chr(65+i), max_sim))

    if similarity_matches:
        similarity_matches.sort(key=lambda x: x[1], reverse=True)
        return similarity_matches[0][0]

    # 4단계: 키워드 기반 고급 매치
    text_keywords = extract_keywords(text)

    keyword_matches = []
    for i, option in enumerate(options):
        if option and str(option).strip():
            option_clean = str(option).strip().strip('"').strip("'")
            option_keywords = extract_keywords(option_clean)

            if not option_keywords:
                continue

            # 키워드 매치 점수 계산
            matched_keywords = set(text_keywords) & set(option_keywords)

            if matched_keywords:
                # 매치 점수: (매치된 키워드 수 / 옵션 키워드 수) * 가중치
                match_ratio = len(matched_keywords) / len(option_keywords)

                # 중요한 키워드일수록 가중치 부여 (길이가 긴 키워드)
                weight = sum(len(kw) for kw in matched_keywords) / sum(len(kw) for kw in option_keywords)

                final_score = match_ratio * weight

                if final_score >= 0.2:  # 30% 이상 매치
                    keyword_matches.append((chr(65+i), final_score))

    if keyword_matches:
        keyword_matches.sort(key=lambda x: x[1], reverse=True)
        return keyword_matches[0][0]

# 간단한 처리 함수
def process_submission(submission_csv, qa_csv):
    """
    제출 파일의 답변에서 정답을 추출하는 함수
    """
    print("🚀 정답 추출 시작...")

    # 파일 로드
    submission_df = pd.read_csv(submission_csv)
    qa_df = pd.read_csv(qa_csv)

    # 병합
    merged_df = submission_df.merge(qa_df, on='ID', how='left')

    # 정답 추출
    def extract_for_row(row):
        model_answer = str(row['answer']) if pd.notna(row['answer']) else ""
        options = [
            row.get('A', ''),
            row.get('B', ''),
            row.get('C', ''),
            row.get('D', '')
        ]
        question = row.get('Question', '')
        return advanced_extract_answer(model_answer, options, question)

    merged_df['extracted_answer'] = merged_df.apply(extract_for_row, axis=1)

    # 결과 저장
    result_df = submission_df.copy()
    result_df['answer'] = merged_df['extracted_answer']

    # 간단한 통계
    total = len(result_df)
    success = result_df['answer'].notna().sum()
    success_rate = success / total * 100

    print(f"📊 처리 결과:")
    print(f"총 항목 수: {total}")
    print(f"✅ 성공적으로 추출: {success}")
    print(f"❌ 추출 실패: {total - success}")
    print(f"📈 성공률: {success_rate:.2f}%")

    return result_df

# 사용 예시
if __name__ == "__main__":
    result_df = process_submission(
        submission_csv=SUBMISSION,
        qa_csv=TEST_CSV_PATH
    )

    # 결과 저장
    result_df.to_csv(SUBMISSION, index=False)
    print("✅ 결과 저장 완료: ./outputs/submission.csv")

