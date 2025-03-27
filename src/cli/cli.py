from typing import Optional, List
import os
from src.utils.file_manager import find_media_files, check_if_processed
from config import INPUT_DIR

def select_operation_mode() -> int:
    """Выбор режима работы программы"""
    print("\n[МЕНЮ] Выберите режим работы:")
    print(" 1. Обработать один файл")
    print(" 2. Обработать все файлы в папке input")
    
    try:
        choice = int(input("[ВВОД] Введите номер выбранного режима (или Enter для режима 1): ") or "1")
        if choice not in [1, 2]:
            print("[ПРЕДУПРЕЖДЕНИЕ] Неверный выбор. Используется режим 1.")
            return 1
        return choice
    except ValueError:
        print("[ПРЕДУПРЕЖДЕНИЕ] Неверный ввод. Используется режим 1.")
        return 1

def select_media_file() -> Optional[str]:
    """Выбор медиа-файла для обработки"""
    media_files = find_media_files()
    
    if not media_files:
        print("[ОШИБКА] Аудио или видео файлы не найдены в директории input.")
        return None
    
    if len(media_files) == 1:
        selected_file = media_files[0]
        is_processed = check_if_processed(os.path.join(INPUT_DIR, selected_file))
        status = " [обработан]" if is_processed else ""
        print(f"[INFO] Найден медиа файл: {selected_file}{status}")
    else:
        print("[INFO] Найдено несколько медиа файлов:")
        for i, file in enumerate(media_files, 1):
            is_processed = check_if_processed(os.path.join(INPUT_DIR, file))
            status = " [обработан]" if is_processed else ""
            print(f" {i}. {file}{status}")
            
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
    
    return os.path.join(INPUT_DIR, selected_file)

def print_processing_result(result, file_name: str):
    """Выводит результаты обработки файла"""
    if result.success:
        print(f"[INFO] Обработка завершена для {file_name}")
        print(f"Результат: {os.path.abspath(result.file_path)}")
        print("Время выполнения операций:")
        for stage, time in result.execution_times.items():
            print(f" - {stage}: {time}")
    else:
        print(f"[ОШИБКА] Ошибка обработки: {result.error}")

def print_batch_statistics(total: int, processed: int, skipped: int, errors: int):
    """Выводит статистику пакетной обработки"""
    print(f"\n[ИТОГ] Обработано файлов: {total}")
    print(f" - Успешно: {processed}")
    print(f" - Пропущено: {skipped}")
    print(f" - Ошибок: {errors}")