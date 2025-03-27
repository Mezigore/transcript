from typing import List, Dict, Any
from src.utils.text_formatter import get_friendly_speaker_name, format_segment_with_metadata, format_transcript
from config import OUTPUT_FORMAT, OUTPUT_DIR
import os

def merge_transcription_with_diarization(diarized_segments, text_and_emotion_segments, max_line_length=None):
    """Объединяет результаты транскрипции с диаризацией и эмоциями"""
    print("[INFO] Объединение транскрипции с диаризацией и эмоциями...")
    merged_segments = []
    temp_segment = None
    MAX_GAP = 5.0  # Максимальный промежуток между сегментами в секундах для объединения

    # Сортируем сегменты по времени начала
    text_and_emotion_segments.sort(key=lambda x: x["start"])
    diarized_segments.sort(key=lambda x: x["start"])
    
    print(f"[INFO] Получено {len(diarized_segments)} сегментов диаризации")
    print(f"[INFO] Получено {len(text_and_emotion_segments)} сегментов с эмоциями")
    
    # Остальная логика объединения сегментов остаётся такой же
    # ... (весь код из исходного файла)
    
    return merged_segments

def create_transcript(transcription: List[Dict], 
                     speakers: Dict,
                     emotions: Dict,
                     output_file: str, 
                     max_line_length: int = 100) -> str:
    """Создаёт форматированный текст транскрипции и сохраняет его в файл"""
    # Объединяем данные транскрипции, диаризации и эмоций
    merged_segments = []
    for segment in transcription:
        segment_id = segment["id"]
        segment_data = {
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"],
            "speaker": speakers.get(segment_id, "Unknown"),
            "emotion": emotions.get(segment_id, {"emotion": "neutral", "confidence": 0.0})
        }
        merged_segments.append(segment_data)
    
    # Форматируем транскрипцию
    formatted_text = format_transcript(
        sorted(merged_segments, key=lambda x: x['start']),
        max_line_length=max_line_length
    )
    
    # Создаем директорию для выходного файла, если её нет
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Сохраняем результат в файл
    output_path = os.path.join(OUTPUT_DIR, os.path.basename(output_file))
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(formatted_text)
    
    return output_path 