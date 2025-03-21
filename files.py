import os
import pandas as pd
import ffmpeg
import textwrap
from typing import List, Dict, Any, Optional, Set
from tqdm import tqdm
from preprocessor import AudioPreprocessor
from config import FILES, INPUT_DIR, OUTPUT_DIR, OUTPUT_FORMAT

# Словарь для перевода эмоций с английского на русский
EMOTION_TRANSLATIONS = FILES['emotion_translations']


def find_media_files() -> List[str]:
    """
    Находит все аудио и видео файлы в директории input.
    
    Returns:
        List[str]: Список имен аудио и видео файлов
    """
    # Поддерживаемые форматы видео
    video_extensions = FILES['video_extensions']
    # Поддерживаемые форматы аудио
    audio_extensions = FILES['audio_extensions']
    # Все поддерживаемые форматы
    media_extensions = video_extensions + audio_extensions
    
    # Проверяем существование директории input
    input_dir = INPUT_DIR
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        
    media_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and 
                  any(f.lower().endswith(ext) for ext in media_extensions)]
    return media_files


def select_media_file() -> Optional[str]:
    """
    Позволяет пользователю выбрать аудио или видео файл из директории input.
    
    Returns:
        Optional[str]: Путь к выбранному файлу или None, если файлы не найдены
    """
    media_files = find_media_files()
    
    if not media_files:
        print("[ОШИБКА] Аудио или видео файлы не найдены в директории input.")
        return None
    
    if len(media_files) == 1:
        selected_file = media_files[0]
        print(f"[INFO] Найден медиа файл: {selected_file}")
    else:
        print("[INFO] Найдено несколько медиа файлов:")
        for i, file in enumerate(media_files, 1):
            print(f"  {i}. {file}")
        
        try:
            choice = int(input("[ВВОД] Выберите номер файла (или Enter для первого файла): ") or "1")
            if 1 <= choice <= len(media_files):
                selected_file = media_files[choice - 1]
            else:
                print("[ПРЕДУПРЕЖДЕНИЕ] Неверный выбор. Используется первый файл.")
                selected_file = media_files[0]
        except ValueError:
            print("[ПРЕДУПРЕЖДЕНИЕ] Неверный ввод. Используется первый файл.")
            selected_file = media_files[0]
    
    # Возвращаем полный путь к файлу
    return os.path.join(INPUT_DIR, selected_file)


def extract_and_process_audio(input_media: str) -> Optional[str]:
    """
    Извлекает аудио из видеофайла или обрабатывает аудиофайл напрямую.
    
    Args:
        input_media (str): Имя входного аудио или видеофайла
        
    Returns:
        Optional[str]: Путь к обработанному аудиофайлу или None в случае ошибки
    """
    # Получаем имя файла без расширения
    input_name = os.path.splitext(os.path.basename(input_media))[0]
    output_audio = f"audio-input-{input_name}.wav"
    processor = AudioPreprocessor()
    
    # Определяем тип файла по расширению
    file_ext = os.path.splitext(input_media)[1].lower()
    audio_extensions = ['.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a', '.wma']
    is_audio_file = file_ext in audio_extensions
    
    if is_audio_file:
        print(f"[INFO] Обработка аудиофайла {input_media}...")
    else:
        print(f"[INFO] Извлечение аудио из {input_media}...")
    
    # Создаем прогресс-бар для процесса извлечения/обработки аудио
    with tqdm(total=0, desc="Обработка медиа", bar_format='{desc}: {elapsed}') as pbar:
        try:
            # Для аудиофайлов, которые не в формате WAV 16kHz mono, конвертируем их
            # Для видеофайлов извлекаем аудиодорожку
            (
                ffmpeg
                .input(input_media)
                .output(output_audio, acodec='pcm_s16le', ac=1, ar='16k')
                .run(quiet=True, overwrite_output=True)
            )
            
            if is_audio_file:
                pbar.set_description("Обработка аудио завершена")
                print(f"[INFO] Аудио обработано: {output_audio}")
            else:
                pbar.set_description("Извлечение аудио завершено")
                print(f"[INFO] Аудио извлечено: {output_audio}")
                
            output_audio = processor.process_audio(output_audio, remove_silence_flag=False, highpass_filter=True, noise_reduction=False, normalize=True, visualize=False)
            return output_audio
        except ffmpeg.Error as e:
            if is_audio_file:
                pbar.set_description("Ошибка обработки аудио")
                print(f"[ОШИБКА] Не удалось обработать аудио: {e.stderr.decode() if hasattr(e, 'stderr') else str(e)}")
            else:
                pbar.set_description("Ошибка извлечения аудио")
                print(f"[ОШИБКА] Не удалось извлечь аудио: {e.stderr.decode() if hasattr(e, 'stderr') else str(e)}")
            return None
        except Exception as e:
            pbar.set_description("Ошибка обработки медиа")
            print(f"[ОШИБКА] Не удалось обработать медиафайл: {e}")
            return None


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
    # Проверяем существование директории output
    output_dir = OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Получаем только имя файла без пути
    output_filename = os.path.basename(output_file)
    # Формируем полный путь в директории output
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"[INFO] Создание файла в формате разговора: {output_path}")
    
    # Создаем маппинг спикеров на буквы (A, B, C...)
    unique_speakers: Set[str] = set(segment['speaker'] for segment in merged_segments)
    speaker_mapping = {speaker: chr(65 + i) for i, speaker in enumerate(sorted(list(unique_speakers)))}
    
    # Объединяем последовательные сегменты с одинаковым спикером и эмоцией
    combined_segments = []
    current_segment = None
    
    with tqdm(total=len(merged_segments), desc="Объединение сегментов", unit="сегмент") as pbar:
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
    
    # Записываем результаты в файл
    with open(output_path, 'w', encoding='utf-8') as f:
        for segment in combined_segments:
            start_time = segment['start']
            end_time = segment['end']
            speaker = segment['speaker']
            text = segment['text']
            emotion = segment.get('emotion', 'unknown')
            emotion_confidence = segment.get('emotion_confidence', 0.0)
            
            # Переводим эмоцию с английского на русский, если есть перевод
            if emotion in EMOTION_TRANSLATIONS:
                emotion = EMOTION_TRANSLATIONS[emotion]
            
            # Форматируем строку в зависимости от настроек
            if OUTPUT_FORMAT['include_emotions'] and emotion_confidence >= OUTPUT_FORMAT['min_confidence_threshold']:
                if OUTPUT_FORMAT['include_confidence']:
                    line = f"[{start_time:.2f} - {end_time:.2f}] {speaker} ({emotion}, {emotion_confidence:.1f}%): {text}"
                else:
                    line = f"[{start_time:.2f} - {end_time:.2f}] {speaker} ({emotion}): {text}"
            else:
                line = f"[{start_time:.2f} - {end_time:.2f}] {speaker}: {text}"
            
            # Если текст длинный, разбиваем его на несколько строк с отступами
            if len(line) > OUTPUT_FORMAT['wrap_width']:
                wrapped_lines = textwrap.wrap(text, width=OUTPUT_FORMAT['wrap_width'] - 30)  # Учитываем длину префикса
                # Первая строка с полным префиксом
                if OUTPUT_FORMAT['include_emotions'] and emotion_confidence >= OUTPUT_FORMAT['min_confidence_threshold']:
                    if OUTPUT_FORMAT['include_confidence']:
                        f.write(f"[{start_time:.2f} - {end_time:.2f}] {speaker} ({emotion}, {emotion_confidence:.1f}%): {wrapped_lines[0]}\n")
                    else:
                        f.write(f"[{start_time:.2f} - {end_time:.2f}] {speaker} ({emotion}): {wrapped_lines[0]}\n")
                else:
                    f.write(f"[{start_time:.2f} - {end_time:.2f}] {speaker}: {wrapped_lines[0]}\n")
                
                # Остальные строки с отступами
                for wrapped_line in wrapped_lines[1:]:
                    f.write(f"{'':>30}{wrapped_line}\n")
            else:
                f.write(f"{line}\n")
            
            # Добавляем пустую строку между сегментами для лучшей читаемости
            f.write("\n")
    
    print(f"[INFO] Файл в формате разговора создан: {output_path}")
    return output_path
