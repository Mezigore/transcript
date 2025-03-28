from typing import List, Dict, Any, Optional
from src.utils.text_formatter import get_friendly_speaker_name, format_segment_with_metadata, format_transcript, format_segment_text
from config import OUTPUT_FORMAT, OUTPUT_DIR
import os

def merge_transcription_with_diarization(diarized_segments: List[Dict], text_and_emotion_segments: Optional[List[Dict]] = None, max_line_length: Optional[int] = None) -> List[Dict]:
    """Объединяет результаты транскрипции с диаризацией и эмоциями"""
    print("[INFO] Объединение транскрипции с диаризацией и эмоциями...")

    # Если нет сегментов с текстом/эмоциями, возвращаем пустой список
    if text_and_emotion_segments is None or len(text_and_emotion_segments) == 0:
        print("[WARNING] Нет сегментов с текстом/эмоциями для объединения")
        return []

    # Если нет сегментов диаризации, используем сегменты текста/эмоций
    if diarized_segments is None or len(diarized_segments) == 0:
        print("[WARNING] Нет сегментов диаризации для объединения, используем только текст/эмоции")
        return text_and_emotion_segments

    print(f"[INFO] Получено {len(diarized_segments)} сегментов диаризации")
    print(f"[INFO] Получено {len(text_and_emotion_segments)} сегментов с текстом/эмоциями")
    
    # Выводим подробную информацию о сегментах диаризации для отладки
    for i, d in enumerate(diarized_segments):
        print(f"[DEBUG] Диаризация сегмент {i}: {d}")

    # Сортируем сегменты по времени
    diarized_segments.sort(key=lambda x: x["start"])
    text_and_emotion_segments.sort(key=lambda x: x["start"])

    # Создаем словарь для отслеживания использованных сегментов с текстом
    used_text_segments = set()
    
    # Минимальное перекрытие для сопоставления сегментов
    MIN_OVERLAP_RATIO = 0.3  # 30% перекрытия
    
    # Результирующие объединенные сегменты
    merged_segments = []
    
    # Подход 1: Для каждого текстового сегмента найдем соответствующий сегмент диаризации
    # Это исправит проблему, когда несколько текстовых сегментов попадают в один сегмент диаризации
    print("[DEBUG] Используем новый алгоритм сопоставления: для каждого текстового сегмента ищем диаризацию")
    
    for i, text_segment in enumerate(text_and_emotion_segments):
        best_match = None
        best_overlap_duration = 0
        
        text_duration = text_segment['end'] - text_segment['start']
        
        for j, diarized in enumerate(diarized_segments):
            # Проверяем перекрытие
            overlap_start = max(diarized['start'], text_segment['start'])
            overlap_end = min(diarized['end'], text_segment['end'])
            
            if overlap_end > overlap_start:
                overlap_duration = overlap_end - overlap_start
                overlap_ratio = overlap_duration / text_duration
                
                print(f"[DEBUG] Текст. сегмент {i} ({text_segment['start']:.2f}-{text_segment['end']:.2f}) "
                      f"перекрывается с диар. сегментом {j} ({diarized['start']:.2f}-{diarized['end']:.2f}, speaker={diarized['speaker']}): "
                      f"перекрытие={overlap_duration:.2f}с, ratio={overlap_ratio:.2f}")
                
                if overlap_duration > best_overlap_duration:
                    best_overlap_duration = overlap_duration
                    best_match = (j, diarized)
        
        if best_match and best_overlap_duration > 0:
            j, diarized = best_match
            
            # Создаем новый сегмент с текстом из текстового сегмента и спикером из диаризации
            merged_segment = {
                'start': text_segment['start'],
                'end': text_segment['end'],
                'speaker': diarized.get('speaker', 'SPEAKER_00'),
                'text': text_segment.get('text', '')
            }
            
            print(f"[DEBUG] Сопоставлен текст. сегмент {i} ({text_segment['start']:.2f}-{text_segment['end']:.2f}) "
                  f"с диар. сегментом {j} ({diarized['start']:.2f}-{diarized['end']:.2f}, speaker={diarized['speaker']})")
            
            # Добавляем эмоции, если есть (улучшенная обработка для обеих моделей)
            if 'emotion' in text_segment:
                # Модель wavlm с одной эмоцией
                merged_segment['emotion'] = text_segment['emotion']
                # Не забываем копировать уверенность, даже если она низкая
                merged_segment['confidence'] = text_segment.get('confidence', 0)
                print(f"[DEBUG] Добавлена эмоция (wavlm): {merged_segment['emotion']} с уверенностью {merged_segment['confidence']:.1f}%")
            elif 'emotions' in text_segment:
                # Модель wav2vec с несколькими параметрами (A,D,V)
                merged_segment['emotions'] = text_segment['emotions']
                print(f"[DEBUG] Добавлены эмоции (wav2vec): {list(merged_segment['emotions'].keys())}")
            
            # Добавляем сегмент в результат
            if merged_segment['text'].strip():
                merged_segments.append(merged_segment)
        else:
            # Не нашли подходящий сегмент диаризации, используем SPEAKER_00
            if text_segment.get('text', '').strip():
                new_segment = text_segment.copy()
                new_segment['speaker'] = 'SPEAKER_00'
                
                print(f"[DEBUG] Текст. сегмент {i} ({text_segment['start']:.2f}-{text_segment['end']:.2f}) "
                      f"не имеет соответствующего сегмента диаризации, назначен SPEAKER_00")
                
                merged_segments.append(new_segment)
    
    # Сортируем по времени начала
    merged_segments.sort(key=lambda x: x['start'])
    
    # Выводим итоговые объединенные сегменты для отладки
    for i, segment in enumerate(merged_segments):
        print(f"[DEBUG] Итоговый сегмент {i}: {segment['start']:.2f}-{segment['end']:.2f}, speaker={segment['speaker']}, "
              f"текст={segment['text'][:50]}...")
    
    print(f"[INFO] Создано {len(merged_segments)} объединенных сегментов")
    return merged_segments


def create_transcript(transcription: List[Dict], 
                     speakers: Dict,
                     emotions: Dict,
                     output_file: str) -> str:
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
        sorted(merged_segments, key=lambda x: x['start'])
    )
    
    # Создаем директорию для выходного файла, если её нет
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Сохраняем результат в файл
    output_path = os.path.join(OUTPUT_DIR, os.path.basename(output_file))
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(formatted_text)
    
    return output_path 