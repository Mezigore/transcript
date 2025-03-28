#!/usr/bin/env python3
import os
import subprocess
import glob
import argparse
from typing import List

def find_transcript_files() -> List[str]:
    """Находит все файлы транскриптов в директории output"""
    transcript_files = []
    
    # Поиск файлов с "_transcript" или "_transcript.txt" в названии
    patterns = ["output/*_transcript", "output/*_transcript.txt"]
    
    for pattern in patterns:
        files = glob.glob(pattern)
        for file in files:
            # Исключаем директории с результатами, которые уже были созданы
            if os.path.isfile(file):
                transcript_files.append(file)
    
    return sorted(transcript_files)

def process_transcripts(no_timestamps: bool = False) -> None:
    """Обрабатывает все найденные транскрипты"""
    transcript_files = find_transcript_files()
    
    if not transcript_files:
        print("Не найдено файлов транскриптов в директории output.")
        return
    
    print(f"Найдено {len(transcript_files)} файлов транскриптов для обработки.")
    
    for i, file in enumerate(transcript_files, 1):
        print(f"[{i}/{len(transcript_files)}] Обработка: {file}")
        try:
            # Вызов скрипта process_transcript.py для обработки файла
            cmd = ["python3", "process_transcript.py", file]
            if no_timestamps:
                cmd.append("--no-timestamps")
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Ошибка при обработке файла {file}: {e}")
        except Exception as e:
            print(f"Непредвиденная ошибка при обработке файла {file}: {e}")
    
    print("Обработка всех транскриптов завершена.")

def parse_args():
    parser = argparse.ArgumentParser(description='Обработка всех транскриптов в директории output')
    parser.add_argument('--no-timestamps', action='store_true',
                       help='Удалить временные метки из результатов постобработки')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process_transcripts(args.no_timestamps) 