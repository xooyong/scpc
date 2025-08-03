from config import BLIP_IMAGE_CAPTION_LOG, BLIP_VQA_LOG
from PIL import Image, ImageEnhance
from datetime import datetime

class ImageAugmentations:
    def __init__(self):
        pass

    def _original(self, image):
        return image

    def _brightness_up(self, image):
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(1.3)

    def _contrast_up(self, image):
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(1.3)

    def _saturation_up(self, image):
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(1.4)

    def _scale_up(self, image):
        w, h = image.size
        new_size = (int(w * 1.2), int(h * 1.2))
        return image.resize(new_size, Image.Resampling.LANCZOS)

    def _sharpness_up(self, image):
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(1.5)

    def _crop_center(self, image):
        w, h = image.size
        crop_size = int(min(w, h) * 0.8)
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        return image.crop((left, top, left + crop_size, top + crop_size))

    # 5-Crop 함수들 추가
    def _crop_top_left(self, image):
        """좌상단 크롭"""
        W, H = image.size
        cx, cy = W // 2, H // 2
        crop_size = 224
        offset = crop_size // 2
        delta = 80

        return image.crop((cx - offset - delta, cy - offset - delta,
                          cx + offset - delta, cy + offset - delta))

    def _crop_top_right(self, image):
        """우상단 크롭"""
        W, H = image.size
        cx, cy = W // 2, H // 2
        crop_size = 224
        offset = crop_size // 2
        delta = 80

        return image.crop((cx - offset + delta, cy - offset - delta,
                          cx + offset + delta, cy + offset - delta))

    def _crop_bottom_left(self, image):
        """좌하단 크롭"""
        W, H = image.size
        cx, cy = W // 2, H // 2
        crop_size = 224
        offset = crop_size // 2
        delta = 80

        return image.crop((cx - offset - delta, cy - offset + delta,
                          cx + offset - delta, cy + offset + delta))

    def _crop_bottom_right(self, image):
        """우하단 크롭"""
        W, H = image.size
        cx, cy = W // 2, H // 2
        crop_size = 224
        offset = crop_size // 2
        delta = 80

        return image.crop((cx - offset + delta, cy - offset + delta,
                          cx + offset + delta, cy + offset + delta))

    def _crop_center_fixed(self, image):
        """중앙 크롭 (5-crop 버전)"""
        W, H = image.size
        cx, cy = W // 2, H // 2
        crop_size = 224
        offset = crop_size // 2

        return image.crop((cx - offset, cy - offset, cx + offset, cy + offset))

# 4. 캡션 결합 함수
def combine_caption(captions_dict, debug=True):
    """모든 캡션을 결합하여 하나의 캡션으로 생성"""
    try:
        valid_captions = []
        for aug_type, caption in captions_dict.items():
            if caption and caption != "Caption generation failed":
                valid_captions.append(caption)

        if not valid_captions:
            return "No valid captions available."

        unique_sentences = set()
        for caption in valid_captions:
            sentences = caption.split('.')
            for sentence in sentences:
                cleaned_sentence = sentence.strip()
                if cleaned_sentence and len(cleaned_sentence) > 5:
                    unique_sentences.add(cleaned_sentence)

        combined_caption = '. '.join(sorted(unique_sentences))
        if len(combined_caption) > 300:
            sentences = combined_caption.split('.')
            combined_caption = '. '.join(sentences[:5])

        if combined_caption and not combined_caption.endswith('.'):
            combined_caption += '.'

        if debug:
            print(f"📝 결합된 캡션 ({len(valid_captions)}개 원본): {combined_caption}")

        return combined_caption

    except Exception as e:
        print(f"⚠️ 캡션 결합 실패: {e}")
        for caption in captions_dict.values():
            if caption and caption != "Caption generation failed":
                return caption
        return "Caption combination failed."

def combine_vqa(captions_dict, debug=True):
    """모든 VQA 답변을 중복 없이 결합"""
    try:
        # 모든 유효한 답변 수집
        all_answers = []
        for aug_type, answer in captions_dict.items():
            if answer and answer.strip() and answer != "Caption generation failed":
                all_answers.append(answer.strip())
        
        if not all_answers:
            return "No valid answers available."
        
        # 중복 제거를 위해 set 사용
        unique_answers = set(all_answers)
        
        # 정렬해서 쉼표로 결합
        combined_answer = ', '.join(sorted(unique_answers))
        
        # 마지막 쉼표 제거
        combined_answer = combined_answer.rstrip(', ')
        
        if debug:
            print(f"📝 원본 답변들: {all_answers}")
            print(f"📝 중복 제거 후: {sorted(unique_answers)}")
            print(f"📝 최종 결합: '{combined_answer}'")
        
        return combined_answer

    except Exception as e:
        print(f"⚠️ 답변 결합 실패: {e}")
        # 예외 시 첫 번째 유효한 답변 반환
        for answer in captions_dict.values():
            if answer and answer.strip() and answer != "Caption generation failed":
                return answer.strip()
        return "Answer combination failed."
    
def init_blip_caption_log(log_file: str = BLIP_IMAGE_CAPTION_LOG):
    """로그 파일을 초기화합니다 (기존 로그 삭제)."""
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("🔄 BLIP IMAGE CAPTIONING 로그 초기화됨\n" + "=" * 50 + "\n")

def save_blip_caption_log(log_text: str, log_file: str = BLIP_IMAGE_CAPTION_LOG):
    """캡션 로그를 지정된 텍스트 파일에 실시간으로 저장합니다."""
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_text)

def init_vqa_log(log_file: str = BLIP_VQA_LOG):
    """로그 파일을 초기화하고 시작 헤더를 추가합니다."""
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("🤖 BLIP VQA 디버그 로그\n")
            f.write(f"⏰ 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
        print(f"✅ 로그 파일 초기화 완료: {log_file}")
    except Exception as e:
        print(f"⚠️ 로그 파일 초기화 실패: {e}")
        
def save_blip_vqa_log(log_text: str, log_file: str = BLIP_VQA_LOG):
    """지정된 텍스트 파일에 로그 한 블록을 추가 저장합니다 (append)."""
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_text)
    except Exception as e:
        print(f"⚠️ 로그 저장 실패: {e}")
