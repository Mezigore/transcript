from config import OUTPUT_FORMAT

def get_friendly_speaker_name(speaker_label):
    """Преобразует метку спикера в более дружественное имя"""
    try:
        number = int(speaker_label.split('_')[-1])
        return f"{number + 1} Спикер"
    except:
        return speaker_label

def format_segment_text(text, max_line_length=None, indent=" "):
    """Форматирует текст с учетом переноса строк"""
    if not max_line_length:
        return text
        
    import textwrap
    return textwrap.fill(
        text,
        width=max_line_length - len(indent),
        initial_indent=indent,
        subsequent_indent=indent,
        break_long_words=False,
        break_on_hyphens=False
    )

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
    
    print(f"[INFO] Создано {len(merged_segments)} объединенных сегментов")
    return merged_segments

def format_transcript(merged_segments, max_line_length=None):
    """Форматирует объединенные сегменты в текст"""
    return "\n\n".join(segment["formatted"] for segment in merged_segments)
