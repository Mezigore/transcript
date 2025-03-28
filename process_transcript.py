#!/usr/bin/env python3
from typing import Dict, List, Tuple, Optional
import re
import os
import json
import argparse
from dataclasses import dataclass
import sys
import pathlib

@dataclass
class EmotionData:
    arousal: float
    dominance: float
    valence: float
    
    def get_mood_emoji(self) -> str:
        if self.valence > 0.6:
            return "😊"
        elif self.valence < 0.4:
            return "😔"
        else:
            return "😐"
            
    def get_trend_description(self) -> str:
        if self.valence > 0.6:
            return "П"  # Позитивный
        elif self.valence < 0.4:
            return "Н"  # Негативный
        else:
            return "Нт"  # Нейтральный
    
    def get_mood_full_name(self) -> str:
        if self.valence > 0.6:
            return "позитивный"
        elif self.valence < 0.4:
            return "негативный"
        else:
            return "нейтральный"
    
    def get_trend_direction(self, prev_emotion: Optional['EmotionData'] = None) -> Dict[str, str]:
        if prev_emotion is None:
            return {"arousal": "", "dominance": "", "valence": ""}
        
        directions = {}
        directions["arousal"] = "→" if abs(self.arousal - prev_emotion.arousal) < 0.05 else ("↑" if self.arousal > prev_emotion.arousal else "↓")
        directions["dominance"] = "→" if abs(self.dominance - prev_emotion.dominance) < 0.05 else ("↑" if self.dominance > prev_emotion.dominance else "↓")
        directions["valence"] = "→" if abs(self.valence - prev_emotion.valence) < 0.05 else ("↑" if self.valence > prev_emotion.valence else "↓")
        
        return directions


@dataclass
class TranscriptSegment:
    start_time: str
    end_time: str
    speaker: int
    emotion: EmotionData
    text: str
    
    def to_llm_json(self, extreme_compact: bool = True) -> str:
        """Формат JSON для анализа LLM-моделями
        
        Args:
            extreme_compact: Если True, максимально сократит данные
        """
        # Уровень округления в зависимости от режима сжатия
        precision = 1 if extreme_compact else 2
            
        segment_dict = {
            "t": f"{self.start_time}-{self.end_time}",
            "s": self.speaker,
            "e": {
                "a": round(self.emotion.arousal, precision),
                "d": round(self.emotion.dominance, precision),
                "v": round(self.emotion.valence, precision)
            },
            "tx": self.text
        }
        return json.dumps(segment_dict, ensure_ascii=False)
    
    def to_human_readable(self, prev_segment: Optional['TranscriptSegment'] = None, max_line_length: int = 100, include_timestamps: bool = True) -> str:
        """Формат для восприятия человеком"""
        trend = self.emotion.get_trend_description()
        emoji = self.emotion.get_mood_emoji()
        
        if prev_segment is not None:
            prev_trend = prev_segment.emotion.get_trend_description()
            if prev_trend != trend:
                trend_text = f"{prev_trend}→{trend}"
            else:
                trend_text = trend
        else:
            trend_text = trend
            
        # Формирование префикса с/без временных меток
        if include_timestamps:
            prefix = f"[{self.start_time}→{self.end_time}] С{self.speaker} {emoji} [{trend_text}]: "
        else:
            prefix = f"С{self.speaker} {emoji} [{trend_text}]: "
        
        # Вместо обрезки текста делаем перенос на новую строку
        if len(prefix + self.text) > max_line_length:
            prefix_length = len(prefix)
            
            # Создаем отступ для последующих строк
            indent = " " * prefix_length
            
            # Разбиваем текст на слова
            words = self.text.split()
            
            lines = []
            current_line = prefix
            
            # Добавляем слова, создавая новую строку при превышении max_line_length
            for word in words:
                # Проверяем, уместится ли слово в текущую строку
                if len(current_line + word) + 1 <= max_line_length:  # +1 для пробела
                    if current_line == prefix:
                        current_line += word
                    else:
                        current_line += " " + word
                else:
                    # Если строка не пустая, добавляем ее в список
                    if current_line != prefix and current_line != indent:
                        lines.append(current_line)
                    
                    # Начинаем новую строку с отступом
                    current_line = indent + word
            
            # Добавляем последнюю строку
            if current_line != prefix and current_line != indent:
                lines.append(current_line)
                
            return "\n".join(lines)
        else:
            return prefix + self.text


def parse_transcript(input_file: str) -> List[TranscriptSegment]:
    """Парсинг файла транскрипта"""
    segments = []
    # Шаблон для формата с временными метками и эмоциями
    pattern_with_timestamps = r'\[(\d+:\d+) -> (\d+:\d+)\] (\d+) Спикер: \[(.*?)\] (.+)'
    # Шаблон для формата без временных меток и эмоций
    pattern_simple = r'^(\d+) Спикер:(.*)$'
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # Пробуем формат с временными метками
    for line in lines:
        match = re.match(pattern_with_timestamps, line)
        if match:
            start_time, end_time, speaker, emotion_text, text = match.groups()
            
            # Обработка текста эмоций
            emotion_data = {"arousal": 0.5, "dominance": 0.5, "valence": 0.5}
            
            if "энтузиазм" in emotion_text.lower():
                # Формат [энтузиазм (100.0%)]
                enthusiasm_match = re.search(r'\((\d+\.\d+)%\)', emotion_text)
                if enthusiasm_match:
                    try:
                        enthusiasm = float(enthusiasm_match.group(1)) / 100.0
                        emotion_data = {"arousal": 0.5, "dominance": 0.5, "valence": enthusiasm}
                    except ValueError:
                        pass
            else:
                # Формат [A:0.51, D:0.54, V:0.65]
                a_match = re.search(r'A:([\d.]+)', emotion_text)
                d_match = re.search(r'D:([\d.]+)', emotion_text)
                v_match = re.search(r'V:([\d.]+)', emotion_text)
                
                if a_match and d_match and v_match:
                    try:
                        emotion_data = {
                            "arousal": float(a_match.group(1)),
                            "dominance": float(d_match.group(1)),
                            "valence": float(v_match.group(1))
                        }
                    except ValueError:
                        pass
            
            emotion = EmotionData(
                arousal=emotion_data["arousal"],
                dominance=emotion_data["dominance"],
                valence=emotion_data["valence"]
            )
            
            segment = TranscriptSegment(
                start_time=start_time,
                end_time=end_time,
                speaker=int(speaker),
                emotion=emotion,
                text=text.strip()
            )
            segments.append(segment)
    
    # Если не нашли сегменты в формате с временными метками, пробуем простой формат
    if not segments:
        current_speaker = None
        current_text = []
        segment_count = 0
        
        for line_num, line in enumerate(lines):
            simple_match = re.match(pattern_simple, line)
            
            if simple_match:
                # Если у нас есть накопленный текст, сохраняем его как сегмент
                if current_speaker is not None and current_text:
                    # Добавляем предыдущий сегмент
                    end_minute = segment_count + 1
                    segment = TranscriptSegment(
                        start_time=f"{segment_count:02d}:00",
                        end_time=f"{end_minute:02d}:00",
                        speaker=current_speaker,
                        emotion=EmotionData(arousal=0.5, dominance=0.5, valence=0.5),
                        text=" ".join(current_text).strip()
                    )
                    segments.append(segment)
                    segment_count += 1
                
                # Начинаем новый сегмент
                current_speaker = int(simple_match.group(1))
                current_text = [simple_match.group(2).strip()]
            elif current_speaker is not None:
                # Продолжаем текущий сегмент
                current_text.append(line.strip())
        
        # Добавляем последний сегмент, если он есть
        if current_speaker is not None and current_text:
            end_minute = segment_count + 1
            segment = TranscriptSegment(
                start_time=f"{segment_count:02d}:00",
                end_time=f"{end_minute:02d}:00",
                speaker=current_speaker,
                emotion=EmotionData(arousal=0.5, dominance=0.5, valence=0.5),
                text=" ".join(current_text).strip()
            )
            segments.append(segment)
                
    if not segments:
        print(f"Предупреждение: В файле {input_file} не найдено сегментов. Проверьте формат файла.")
    
    return segments


def merge_similar_segments(segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
    """Объединяет подряд идущие сегменты одного спикера с одинаковым настроением"""
    if not segments:
        return []
        
    merged_segments = []
    current_group = [segments[0]]
    
    for i in range(1, len(segments)):
        current = segments[i]
        prev = segments[i-1]
        
        # Проверяем, совпадает ли спикер и настроение
        if (current.speaker == prev.speaker and 
            current.emotion.get_trend_description() == prev.emotion.get_trend_description()):
            current_group.append(current)
        else:
            # Завершаем текущую группу
            if len(current_group) == 1:
                merged_segments.append(current_group[0])
            else:
                start_time = current_group[0].start_time
                end_time = current_group[-1].end_time
                merged_text = " ".join(s.text for s in current_group)
                
                merged_segment = TranscriptSegment(
                    start_time=start_time,
                    end_time=end_time,
                    speaker=current_group[0].speaker,
                    emotion=current_group[0].emotion,
                    text=merged_text
                )
                merged_segments.append(merged_segment)
            
            # Начинаем новую группу
            current_group = [current]
    
    # Обрабатываем последнюю группу
    if len(current_group) == 1:
        merged_segments.append(current_group[0])
    else:
        start_time = current_group[0].start_time
        end_time = current_group[-1].end_time
        merged_text = " ".join(s.text for s in current_group)
        
        merged_segment = TranscriptSegment(
            start_time=start_time,
            end_time=end_time,
            speaker=current_group[0].speaker,
            emotion=current_group[0].emotion,
            text=merged_text
        )
        merged_segments.append(merged_segment)
    
    return merged_segments


def process_transcript(input_file: str, max_line_length: int = 100, include_timestamps: bool = True, extreme_compact: bool = True) -> None:
    """Обработка транскрипта и сохранение в двух форматах
    
    Args:
        input_file: Путь к входному файлу
        max_line_length: Максимальная длина строки в человекочитаемом формате
        include_timestamps: Включать ли временные метки в человекочитаемый формат
        extreme_compact: Максимально сократить данные в JSON (округление до 1 знака)
    """
    segments = parse_transcript(input_file)
    
    # Объединяем похожие сегменты
    merged_segments = merge_similar_segments(segments)
    
    # Получение имени файла без расширения как имя для папки
    file_path = pathlib.Path(input_file)
    output_dir_name = file_path.stem
    output_dir = os.path.join("output", output_dir_name)
    
    # Создание директории для выходных файлов
    os.makedirs(output_dir, exist_ok=True)
    
    # Создание файла для LLM
    llm_output_path = os.path.join(output_dir, "transcript_llm.json")
    with open(llm_output_path, 'w', encoding='utf-8') as f:
        # Добавляем легенду в начало файла
        legend = {
            "_legend": {
                "t": "time (start-end)",
                "s": "speaker number",
                "e": {
                    "a": "arousal (0-1)",
                    "d": "dominance (0-1)",
                    "v": "valence (0-1)"
                },
                "tx": "text content"
            }
        }
        
        f.write("[\n")
        # Записываем легенду как первый элемент массива
        f.write("  " + json.dumps(legend, ensure_ascii=False) + ",\n")
        
        for i, segment in enumerate(segments):  # Используем оригинальные сегменты для LLM
            f.write("  " + segment.to_llm_json(extreme_compact))
            if i < len(segments) - 1:
                f.write(",")
            f.write("\n")
        f.write("]\n")
    
    # Создание файла для человека
    human_output_path = os.path.join(output_dir, "transcript_human.txt")
    with open(human_output_path, 'w', encoding='utf-8') as f:
        # Добавляем легенду
        f.write("# Легенда:\n")
        f.write("# С - Спикер\n")
        f.write("# П - позитивный\n")
        f.write("# Н - негативный\n")
        f.write("# Нт - нейтральный\n")
        f.write("# 😊 - позитивный\n")
        f.write("# 😐 - нейтральный\n")
        f.write("# 😔 - негативный\n")
        f.write("#\n")
        f.write("# Пояснение к эмоциональной модели:\n")
        f.write("# В отличие от упрощенных буквенных обозначений (П/Н/Нт), которые показывают только валентность\n")
        f.write("# (позитивность/негативность), полная AVD-модель учитывает три параметра:\n")
        f.write("#   A (Arousal) - возбуждение/активация эмоции (0 - спокойный, 1 - возбужденный)\n")
        f.write("#   V (Valence) - валентность/знак эмоции (0 - негативный, 1 - позитивный)\n")
        f.write("#   D (Dominance) - доминантность/контроль (0 - подчиненный, 1 - доминирующий)\n")
        f.write("# Эмодзи и буквы отражают только валентность (V), в то время как полная картина эмоций\n")
        f.write("# представлена в файле transcript_llm.json со всеми тремя параметрами AVD.\n")
        f.write("\n")
        
        prev_segment = None
        for segment in merged_segments:
            f.write(segment.to_human_readable(prev_segment, max_line_length, include_timestamps) + "\n")
            prev_segment = segment

    print(f"Обработка завершена. Результаты сохранены в папке {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description='Обработка транскрипта и форматирование в разные форматы')
    parser.add_argument('input_file', nargs='?', default="output/test_transcript.txt",
                       help='Путь к входному файлу транскрипта')
    parser.add_argument('--no-timestamps', action='store_true',
                       help='Не включать временные метки в результаты постобработки')
    parser.add_argument('--max-line-length', type=int, default=100,
                       help='Максимальная длина строки в результирующем файле')
    parser.add_argument('--no-extreme-compact', action='store_true',
                       help='Отключить экстремальное сжатие данных (больше цифр после запятой)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Ошибка: файл {args.input_file} не найден")
    else:
        process_transcript(
            args.input_file, 
            args.max_line_length, 
            not args.no_timestamps,
            not args.no_extreme_compact
        ) 