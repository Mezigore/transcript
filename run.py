import os
import shutil
import time  
from src.files import *
from src.transcribe import transcribe_audio  
from src.diarization import diarize
from src.emotion import analyze_emotions
from config import API_KEYS, TEMP_DIR
from src.combine import merge_transcription_with_diarization
from typing import Dict, Union

def format_time(seconds: float) -> str:
    """Форматирует время в минуты и секунды."""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes}м {seconds}с"

def process_media(media_path: str, hf_token: Union[str, None] = None) -> Dict[str, Union[bool, str]]:
    start_time = time.time()
    execution_times = {}

    # Validate input path
    if not os.path.exists(media_path):
        return {"success": False, "error": "[ОШИБКА] Файл медиа не найден"}
    
    if hf_token is None:
        hf_token = API_KEYS['huggingface']
    
    # Audio extraction timing
    audio_start = time.time()
    audio_result = extract_and_process_audio(media_path)
    audio_path = audio_result
    execution_times['audio_extraction'] = format_time(time.time() - audio_start)
    
    if audio_path is None:
        return {"success": False, "error": "[ОШИБКА] Не удалось извлечь аудио"}
    
    # Transcription timing
    try:
        transcription_start = time.time()
        transcription_segments = transcribe_audio(audio_path)
        execution_times['transcription'] = format_time(time.time() - transcription_start)
    except Exception as e:
        return {"success": False, "error": f"[ОШИБКА] Транскрипция не удалась: {str(e)}"}
    
    # Diarization timing
    try:
        diar_start = time.time()
        diarized_segments = diarize(audio_path, hf_token)
        execution_times['diarization'] = format_time(time.time() - diar_start)
    except Exception as e:
        return {"success": False, "error": f"[ОШИБКА] Диаризация не удалась: {str(e)}"}
    
    # Emotion analysis timing
    try:
        emotion_start = time.time()
        text_and_emotion_segments = analyze_emotions(audio_path, transcription_segments)
        execution_times['emotion_analysis'] = format_time(time.time() - emotion_start)
    except Exception as e:
        return {"success": False, "error": f"[ОШИБКА] Анализ эмоций не удался: {str(e)}"}
    
    # Merging results timing
    merge_start = time.time()
    merged_segments = merge_transcription_with_diarization(diarized_segments, text_and_emotion_segments, max_line_length=100)
    execution_times['merging'] = format_time(time.time() - merge_start)
    
    # File creation timing
    base_output = os.path.splitext(os.path.basename(media_path))[0]
    output_conversation = f"{base_output}_transcript.txt"
    
    file_creation_start = time.time()
    conversation_file = create_conversation_file(merged_segments, output_conversation, max_line_length=100)
    execution_times['file_creation'] = format_time(time.time() - file_creation_start)
    
    # Cleanup
    # Check if audio_path is a tuple (path, cleanup_flag)
    if isinstance(audio_path, tuple) and len(audio_path) == 2:
        audio_file, cleanup_flag = audio_path
        if cleanup_flag and os.path.exists(audio_file):
            os.remove(audio_file)
            print(f"[INFO] Удален обработанный аудиофайл: {audio_file}")
    elif os.path.exists(audio_path):
        os.remove(audio_path)
    
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    
    total_time = time.time() - start_time
    execution_times['total'] = format_time(total_time)
    
    return {
        "success": True, 
        "file_path": conversation_file,
        "execution_times": execution_times
    }

if __name__ == "__main__":
    # Загрузка токена из конфигурации
    hf_token = API_KEYS['huggingface']
    
    # Выбор режима работы
    mode = select_operation_mode()
    
    if mode == 1:
        # Режим обработки одного файла
        media_path = select_media_file()
        
        if not media_path:
            print("[ОШИБКА] Файл не выбран. Поместите медиафайлы в директорию 'input'.")
            exit(1)
        
        # Проверка, был ли файл уже обработан
        if check_if_processed(media_path):
            should_process = input("[ВВОД] Этот файл уже был обработан. Обработать снова? (y/n, по умолчанию n): ").lower() == 'y'
            if not should_process:
                print("[INFO] Обработка файла отменена пользователем.")
                exit(0)
        
        # Обработка медиафайла
        result = process_media(media_path)
        
        if result["success"]:
            print(f"[INFO] Обработка завершена. Результат сохранен в: {result['file_path']}")
            print(f"[INFO] Полный путь: {os.path.abspath(str(result['file_path']))}")
            print("\n[INFO] Время выполнения операций:")
            print(f" - Извлечение аудио: {result['execution_times']['audio_extraction']}")
            print(f" - Транскрипция: {result['execution_times']['transcription']}")
            print(f" - Диаризация: {result['execution_times']['diarization']}")
            print(f" - Анализ эмоций: {result['execution_times']['emotion_analysis']}")
            print(f" - Объединение результатов: {result['execution_times']['merging']}")
            print(f" - Создание файла: {result['execution_times']['file_creation']}")
            print(f" - Общее время: {result['execution_times']['total']}")
        else:
            print(f"[ОШИБКА] Ошибка обработки медиафайла: {result['error']}")
    
    else:
        # Режим обработки всех файлов
        media_files = find_media_files()
        
        if not media_files:
            print("[ОШИБКА] Медиафайлы не найдены в директории input.")
            exit(1)
        
        # Отображаем информацию о всех файлах
        print("[INFO] Найдены следующие медиафайлы:")
        for i, file in enumerate(media_files, 1):
            full_path = os.path.join(INPUT_DIR, file)
            is_processed = check_if_processed(full_path)
            status = " [обработан]" if is_processed else ""
            print(f" {i}. {file}{status}")
        
        # Предлагаем пропустить уже обработанные файлы
        skip_processed = input("[ВВОД] Пропустить уже обработанные файлы? (y/n, по умолчанию y): ").lower() != 'n'
        
        # Счетчики для статистики
        processed_count = 0
        error_count = 0
        skipped_count = 0
        
        # Обрабатываем каждый файл
        for file in media_files:
            full_path = os.path.join(INPUT_DIR, file)
            is_processed = check_if_processed(full_path)
            
            # Пропускаем уже обработанные файлы, если пользователь указал это
            if skip_processed and is_processed:
                print(f"[INFO] Пропуск уже обработанного файла: {file}")
                skipped_count += 1
                continue
            
            print(f"\n[INFO] Обработка файла: {file}")
            
            # Обработка медиафайла
            result = process_media(full_path)
            
            if result["success"]:
                print(f"[INFO] Обработка завершена. Результат сохранен в: {result['file_path']}")
                print(f"[INFO] Полный путь: {os.path.abspath(str(result['file_path']))}")
                print("\n[INFO] Время выполнения операций:")
                print(f" - Извлечение аудио: {result['execution_times']['audio_extraction']}")
                print(f" - Транскрипция: {result['execution_times']['transcription']}")
                print(f" - Диаризация: {result['execution_times']['diarization']}")
                print(f" - Анализ эмоций: {result['execution_times']['emotion_analysis']}")
                print(f" - Объединение результатов: {result['execution_times']['merging']}")
                print(f" - Создание файла: {result['execution_times']['file_creation']}")
                print(f" - Общее время: {result['execution_times']['total']}")
                processed_count += 1
            else:
                print(f"[ОШИБКА] Ошибка обработки медиафайла: {result['error']}")
                error_count += 1
        
        # Отображаем итоговую статистику
        print(f"\n[INFO] Итоговая статистика обработки:")
        print(f" - Всего файлов: {len(media_files)}")
        print(f" - Успешно обработано: {processed_count}")
        print(f" - Пропущено (уже обработанные): {skipped_count}")
        print(f" - Ошибки обработки: {error_count}")
