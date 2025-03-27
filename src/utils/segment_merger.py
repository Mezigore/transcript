from typing import List, Dict, Any, Optional
from src.utils.text_formatter import get_friendly_speaker_name, format_segment_with_metadata, format_transcript, format_segment_text
from config import OUTPUT_FORMAT, OUTPUT_DIR
import os

def merge_transcription_with_diarization(diarized_segments: List[Dict], text_and_emotion_segments: Optional[List[Dict]] = None, max_line_length: Optional[int] = None) -> List[Dict]:
    """Объединяет результаты транскрипции с диаризацией и эмоциями"""
    print("[INFO] Объединение транскрипции с диаризацией и эмоциями...")
    merged_segments = []
    temp_segment = None
    MAX_GAP = 5.0  # Максимальный промежуток между сегментами в секундах для объединения

    # Сортируем сегменты по времени начала
    if text_and_emotion_segments is not None and len(text_and_emotion_segments) > 0:
        text_and_emotion_segments.sort(key=lambda x: x["start"])
        print(f"[INFO] Получено {len(text_and_emotion_segments)} сегментов с эмоциями")
    else:
        text_and_emotion_segments = []
        print("[INFO] Анализ эмоций пропущен или не вернул результатов")
    diarized_segments.sort(key=lambda x: x["start"])
    
    print(f"[INFO] Получено {len(diarized_segments)} сегментов диаризации")

    for whisper_segment in text_and_emotion_segments:
        whisper_start = float(whisper_segment["start"])
        whisper_end = float(whisper_segment["end"])
        
        # Находим спикера с максимальным перекрытием
        max_overlap = 0
        best_match = None
        
        for speaker_segment in diarized_segments:
            if speaker_segment["end"] < whisper_start or speaker_segment["start"] > whisper_end:
                continue
                
            overlap_start = max(whisper_start, float(speaker_segment["start"]))
            overlap_end = min(whisper_end, float(speaker_segment["end"]))
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = speaker_segment
        
        # Определяем спикера
        if best_match:
            speaker = get_friendly_speaker_name(best_match["speaker"])
        elif merged_segments:
            speaker = merged_segments[-1]["speaker"]
        else:
            speaker = "Неизвестный спикер"
        
        # Обрабатываем эмоции
        emotion = whisper_segment.get("emotion", "")
        emotion_confidence = whisper_segment.get("confidence", 0.0)
        
        if emotion_confidence < OUTPUT_FORMAT['min_confidence_threshold']:
            emotion = ""
            emotion_confidence = 0.0
        
        # Обработка сегментов
        if temp_segment is None:
            # Первый сегмент
            temp_segment = {
                "start": whisper_start,
                "end": whisper_end,
                "text": whisper_segment["text"].strip(),
                "speaker": speaker,
                "emotion": emotion,
                "confidence_sum": emotion_confidence,
                "segments_count": 1
            }
        elif (temp_segment["speaker"] == speaker and 
              temp_segment["emotion"] == emotion and 
              whisper_start - temp_segment["end"] <= MAX_GAP):
            # Объединяем с текущим сегментом
            temp_segment["end"] = whisper_end
            
            # Добавляем пробел при необходимости
            separator = " " if temp_segment["text"] and temp_segment["text"][-1] not in ".!?" else ""
            temp_segment["text"] += separator + whisper_segment["text"].strip()
            
            temp_segment["confidence_sum"] += emotion_confidence
            temp_segment["segments_count"] += 1
        else:
            # Завершаем текущий сегмент и начинаем новый
            avg_confidence = temp_segment["confidence_sum"] / temp_segment["segments_count"]
            
            start_time = f"{int(temp_segment['start'] // 60)}:{int(temp_segment['start'] % 60):02d}"
            end_time = f"{int(temp_segment['end'] // 60)}:{int(temp_segment['end'] % 60):02d}"
            
            timestamp_info = f"[{start_time} - {end_time}] " if OUTPUT_FORMAT['include_timestamps'] else ""
            emotion_info = f" ({temp_segment['emotion']}, {min(avg_confidence, 100):.1f}%)" if avg_confidence >= OUTPUT_FORMAT['min_confidence_threshold'] else ""
            
            header = f"{timestamp_info}{temp_segment['speaker']}{emotion_info}:"
            formatted_text = format_segment_text(temp_segment['text'], max_line_length=max_line_length)
            formatted_segment = f"{header}\n{formatted_text}"
            
            merged_segments.append({
                "start": temp_segment["start"],
                "end": temp_segment["end"],
                "text": temp_segment["text"],
                "speaker": temp_segment["speaker"],
                "emotion": temp_segment["emotion"],
                "emotion_confidence": avg_confidence,
                "formatted": formatted_segment
            })
            
            # Новый сегмент
            temp_segment = {
                "start": whisper_start,
                "end": whisper_end,
                "text": whisper_segment["text"].strip(),
                "speaker": speaker,
                "emotion": emotion,
                "confidence_sum": emotion_confidence,
                "segments_count": 1
            }
    
    # Добавляем последний сегмент
    if temp_segment:
        avg_confidence = temp_segment["confidence_sum"] / temp_segment["segments_count"]
        
        start_time = f"{int(temp_segment['start'] // 60)}:{int(temp_segment['start'] % 60):02d}"
        end_time = f"{int(temp_segment['end'] // 60)}:{int(temp_segment['end'] % 60):02d}"
        
        timestamp_info = f"[{start_time} - {end_time}] " if OUTPUT_FORMAT['include_timestamps'] else ""
        emotion_info = f" ({temp_segment['emotion']}, {min(avg_confidence, 100):.1f}%)" if avg_confidence >= OUTPUT_FORMAT['min_confidence_threshold'] else ""
        
        header = f"{timestamp_info}{temp_segment['speaker']}{emotion_info}:"
        formatted_text = format_segment_text(temp_segment['text'], max_line_length=max_line_length)
        formatted_segment = f"{header}\n{formatted_text}"
        
        merged_segments.append({
            "start": temp_segment["start"],
            "end": temp_segment["end"],
            "text": temp_segment["text"],
            "speaker": temp_segment["speaker"],
            "emotion": temp_segment["emotion"],
            "emotion_confidence": avg_confidence,
            "formatted": formatted_segment
        })
    
    # Если нет сегментов с эмоциями, создаем сегменты только на основе диаризации
    if not text_and_emotion_segments and diarized_segments:
        print("[INFO] Создание сегментов только на основе диаризации")
        
        # Группируем последовательные сегменты одного спикера
        current_speaker = None
        speaker_segment = None
        
        for segment in diarized_segments:
            speaker = get_friendly_speaker_name(segment["speaker"])
            
            if current_speaker is None or current_speaker != speaker:
                # Сохраняем предыдущий сегмент
                if speaker_segment is not None:
                    timestamp_info = f"[{int(speaker_segment['start'] // 60)}:{int(speaker_segment['start'] % 60):02d} - {int(speaker_segment['end'] // 60)}:{int(speaker_segment['end'] % 60):02d}] " if OUTPUT_FORMAT['include_timestamps'] else ""
                    header = f"{timestamp_info}{speaker_segment['speaker']}:"
                    formatted_segment = f"{header}\n {speaker_segment.get('text', '(Без транскрипции)')}"
                    
                    merged_segments.append({
                        "start": speaker_segment["start"],
                        "end": speaker_segment["end"],
                        "text": speaker_segment.get('text', '(Без транскрипции)'),
                        "speaker": speaker_segment["speaker"],
                        "emotion": "",
                        "emotion_confidence": 0.0,
                        "formatted": formatted_segment
                    })
                
                # Новый сегмент
                speaker_segment = {
                    "start": segment["start"],
                    "end": segment["end"],
                    "speaker": speaker
                }
                current_speaker = speaker
            else:
                # Продолжаем текущий сегмент
                if speaker_segment is not None:
                    speaker_segment["end"] = segment["end"]
        
        # Добавляем последний сегмент
        if speaker_segment is not None:
            timestamp_info = f"[{int(speaker_segment['start'] // 60)}:{int(speaker_segment['start'] % 60):02d} - {int(speaker_segment['end'] // 60)}:{int(speaker_segment['end'] % 60):02d}] " if OUTPUT_FORMAT['include_timestamps'] else ""
            header = f"{timestamp_info}{speaker_segment['speaker']}:"
            formatted_segment = f"{header}\n {speaker_segment.get('text', '(Без транскрипции)')}"
            
            merged_segments.append({
                "start": speaker_segment["start"],
                "end": speaker_segment["end"],
                "text": speaker_segment.get('text', '(Без транскрипции)'),
                "speaker": speaker_segment["speaker"],
                "emotion": "",
                "emotion_confidence": 0.0,
                "formatted": formatted_segment
            })
    
    print(f"[INFO] Создано {len(merged_segments)} объединенных сегментов")
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