import os
import pandas as pd
import ffmpeg
import textwrap
from typing import List, Dict, Any, Optional, Set
from tqdm import tqdm
from preprocessor import AudioPreprocessor

# Словарь для перевода эмоций с английского на русский
EMOTION_TRANSLATIONS = {
    "neutral": "нейтральность",
    "happiness": "радость",
    "positive": "радость",
    "sadness": "грусть",
    "sad": "грусть",
    "anger": "гнев",
    "angry": "гнев",
    "злость": "гнев",
    "fear": "страх",
    "disgust": "отвращение",
    "surprise": "удивление",
    "enthusiasm": "энтузиазм",
    "disappointment": "разочарование",
    "other": "другое",
    "unknown": "неизвестно"
}


def find_mp4_files() -> List[str]:
    """
    Находит все MP4 файлы в текущей директории.
    
    Returns:
        List[str]: Список имен MP4 файлов
    """
    mp4_files = [f for f in os.listdir('.') if f.lower().endswith('.mp4')]
    return mp4_files


def select_mp4_file() -> Optional[str]:
    """
    Позволяет пользователю выбрать MP4 файл из текущей директории.
    
    Returns:
        Optional[str]: Имя выбранного файла или None, если файлы не найдены
    """
    mp4_files = find_mp4_files()
    
    if not mp4_files:
        print("Ошибка: MP4 файлы не найдены в текущей директории.")
        return None
    
    if len(mp4_files) == 1:
        selected_file = mp4_files[0]
        print(f"Найден один MP4 файл: {selected_file}")
    else:
        print("Найдено несколько MP4 файлов:")
        for i, file in enumerate(mp4_files, 1):
            print(f"{i}. {file}")
        
        try:
            choice = int(input("Выберите номер файла для обработки (или нажмите Enter для выбора первого файла): ") or "1")
            if 1 <= choice <= len(mp4_files):
                selected_file = mp4_files[choice - 1]
            else:
                print("Неверный выбор. Используется первый файл.")
                selected_file = mp4_files[0]
        except ValueError:
            print("Неверный ввод. Используется первый файл.")
            selected_file = mp4_files[0]
    
    return selected_file


def extract_and_process_audio(input_video: str) -> Optional[str]:
    """
    Извлекает аудио из видеофайла и выполняет его предобработку.
    
    Args:
        input_video (str): Имя входного видеофайла
        
    Returns:
        Optional[str]: Путь к обработанному аудиофайлу или None в случае ошибки
    """
    input_video_name = input_video.replace(".mp4", "")
    output_audio = "audio-input.wav"
    processor = AudioPreprocessor()
    
    print(f"Извлечение аудио из {input_video}...")
    
    # Создаем прогресс-бар для процесса извлечения аудио
    with tqdm(total=0, desc="Извлечение аудио", bar_format='{desc}: {elapsed}') as pbar:
        try:
            (
                ffmpeg
                .input(input_video)
                .output(output_audio, acodec='pcm_s16le', ac=1, ar='16k')
                .run(quiet=True, overwrite_output=True)
            )
            pbar.set_description("Извлечение аудио завершено")
            print(f"\nАудио успешно извлечено: {output_audio}")
            output_audio = processor.process(output_audio, visualize=False)
            return output_audio
        except ffmpeg.Error as e:
            pbar.set_description("Ошибка извлечения аудио")
            print(f"\nОшибка при извлечении аудио: {e.stderr.decode() if hasattr(e, 'stderr') else str(e)}")
            return None
        except Exception as e:
            pbar.set_description("Ошибка обработки аудио")
            print(f"\nОшибка при обработке аудио: {e}")
            return None


def create_csv_file(merged_segments: List[Dict[str, Any]], output_file: str) -> str:
    """
    Создает CSV файл из сегментов транскрипции.
    
    Args:
        merged_segments (List[Dict[str, Any]]): Список сегментов с транскрипцией
        output_file (str): Имя выходного файла
        
    Returns:
        str: Путь к созданному CSV файлу
    """
    print(f"Создание CSV файла: {output_file}")
    
    # Создаем DataFrame из сегментов
    df = pd.DataFrame(merged_segments)
    
    # Убедимся, что все колонки присутствуют и в правильном порядке
    columns = ['start', 'end', 'speaker', 'emotion', 'text']
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    
    # Переупорядочиваем колонки
    df = df[columns]
    
    # Форматируем временные метки для лучшей читаемости с прогресс-баром
    with tqdm(total=2, desc="Форматирование данных CSV", unit="шаг") as pbar:
        df['start'] = df['start'].apply(lambda x: f"{float(x):.2f}")
        pbar.update(1)
        
        df['end'] = df['end'].apply(lambda x: f"{float(x):.2f}")
        pbar.update(1)
    
    # Сохраняем в CSV с правильной кодировкой и обработкой специальных символов
    with tqdm(total=1, desc="Сохранение CSV файла", unit="файл") as pbar:
        df.to_csv(output_file, index=False, encoding='utf-8-sig', quoting=1)
        pbar.update(1)
    
    print(f"CSV файл успешно создан: {output_file}")
    return output_file


def create_conversation_file(merged_segments: List[Dict[str, Any]], output_file: str) -> str:
    """
    Создает файл в формате разговора (conversation format) из сегментов транскрипции.
    Формат: [start_time - end_time] Speaker (emotion): text
    
    Args:
        merged_segments (List[Dict[str, Any]]): Список сегментов с транскрипцией
        output_file (str): Имя выходного файла
        
    Returns:
        str: Путь к созданному файлу разговора
    """
    print(f"Создание файла в формате разговора: {output_file}")
    
    # Создаем маппинг спикеров на буквы (A, B, C...)
    unique_speakers: Set[str] = set(segment['speaker'] for segment in merged_segments)
    speaker_mapping = {speaker: chr(65 + i) for i, speaker in enumerate(sorted(list(unique_speakers)))}
    
    # Объединяем последовательные сегменты с одинаковым спикером и эмоцией
    combined_segments = []
    current_segment = None
    
    with tqdm(total=len(merged_segments), desc="Объединение последовательных сегментов", unit="сегмент") as pbar:
        for segment in merged_segments:
            # Заменяем SPEAKER_XX на соответствующую букву
            segment['speaker'] = speaker_mapping[segment['speaker']]
            
            if current_segment is None:
                current_segment = segment.copy()
            elif (segment['speaker'] == current_segment['speaker'] and 
                  segment.get('emotion', 'unknown') == current_segment.get('emotion', 'unknown')):
                # Объединяем сегменты с тем же спикером и эмоцией
                current_segment['end'] = segment['end']
                current_segment['text'] += " " + segment['text']
                # Если есть уверенность в эмоции, берем среднее значение
                if 'emotion_confidence' in segment and 'emotion_confidence' in current_segment:
                    current_segment['emotion_confidence'] = (current_segment['emotion_confidence'] + segment['emotion_confidence']) / 2
            else:
                combined_segments.append(current_segment)
                current_segment = segment.copy()
            
            pbar.update(1)
    
    # Добавляем последний сегмент
    if current_segment is not None:
        combined_segments.append(current_segment)
    
    # Формируем строки в формате [start - end] Speaker (emotion confidence%): с переносами строк
    conversation_lines = []
    
    with tqdm(total=len(combined_segments), desc="Форматирование сегментов разговора", unit="сегмент") as pbar:
        for segment in combined_segments:
            start_time = f"{float(segment['start']):.2f}"
            end_time = f"{float(segment['end']):.2f}"
            speaker = segment['speaker']
            emotion_eng = segment.get('emotion', 'unknown')
            # Переводим эмоцию на русский язык
            emotion_rus = EMOTION_TRANSLATIONS.get(emotion_eng.lower(), emotion_eng)
            
            # Добавляем процент уверенности, если он доступен
            confidence = segment.get('emotion_confidence', 0)
            if confidence > 0:
                emotion_display = f"{emotion_rus} {confidence:.0f}%"
            else:
                emotion_display = emotion_rus
                
            text = segment['text']
            
            # Создаем заголовок сегмента с переведенной эмоцией и уверенностью
            header = f"[{start_time} - {end_time}] {speaker} ({emotion_display}):"
            
            # Добавляем заголовок как отдельную строку
            conversation_lines.append(header)
            
            # Форматируем текст с переносами строк (ширина 80 символов)
            wrapped_lines = textwrap.wrap(text, width=80, initial_indent='    ', subsequent_indent='    ')
            
            # Добавляем текст с отступом
            conversation_lines.extend(wrapped_lines)
            
            # Добавляем пустую строку после сегмента для лучшей читаемости
            conversation_lines.append("")
            
            pbar.update(1)
    
    # Записываем в файл
    output_file = output_file.replace('.csv', '.txt')
    with tqdm(total=1, desc="Сохранение файла разговора", unit="файл") as pbar:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(conversation_lines))
        pbar.update(1)
    
    print(f"Файл разговора успешно создан: {output_file}")
    return output_file
