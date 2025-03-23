import os
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
import noisereduce as nr
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Union
from config import AUDIO_PREPROCESSOR


class AudioPreprocessor:
    def __init__(self, target_sr: int = None, mono: bool = None):
        """
        Инициализация препроцессора аудио
        
        Args:
            target_sr (int): Целевая частота дискретизации (по умолчанию из конфига)
            mono (bool): Преобразовывать ли в моно (по умолчанию из конфига)
        """
        self.target_sr = target_sr if target_sr is not None else AUDIO_PREPROCESSOR['target_sr']
        self.mono = mono if mono is not None else AUDIO_PREPROCESSOR['mono']
    
    def load_audio(self, file_path: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """
        Загрузка аудиофайла
        
        Args:
            file_path (str): Путь к аудиофайлу
            
        Returns:
            tuple: аудиоданные и частота дискретизации
        """
        print(f"[INFO] Загрузка аудио: {file_path}")
        try:
            audio, sr = librosa.load(file_path, sr=None, mono=self.mono)
            return audio, sr
        except Exception as e:
            print(f"[ОШИБКА] Не удалось загрузить аудио: {e}")
            return None, None
    
    def resample(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        """
        Ресемплинг аудио до целевой частоты дискретизации
        
        Args:
            audio (np.ndarray): Аудиоданные
            orig_sr (int): Исходная частота дискретизации
            
        Returns:
            np.ndarray: Ресемплированное аудио
        """
        if orig_sr != self.target_sr:
            print(f"[INFO] Ресемплинг с {orig_sr}Гц до {self.target_sr}Гц")
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=self.target_sr)
        return audio
    
    def convert_to_mono(self, audio: np.ndarray) -> np.ndarray:
        """
        Преобразование стерео в моно
        
        Args:
            audio (np.ndarray): Аудиоданные
            
        Returns:
            np.ndarray: Моно аудио
        """
        if len(audio.shape) > 1 and audio.shape[0] == 2:
            print("[INFO] Преобразование стерео в моно")
            return np.mean(audio, axis=0)
        return audio
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Нормализация амплитуды аудио
        
        Args:
            audio (np.ndarray): Аудиоданные
            
        Returns:
            np.ndarray: Нормализованное аудио
        """
        print("[INFO] Нормализация амплитуды")
        max_amplitude = np.max(np.abs(audio))
        if max_amplitude > 0:
            return audio / max_amplitude
        return audio
    
    def remove_silence(self, audio: np.ndarray, top_db: int = None) -> np.ndarray:
        """
        Удаление тишины из аудио
        
        Args:
            audio (np.ndarray): Аудиоданные
            top_db (int): Порог в дБ для определения тишины
            
        Returns:
            np.ndarray: Аудио без тишины
        """
        if top_db is None:
            top_db = AUDIO_PREPROCESSOR['silence_removal']['top_db']
            
        print(f"[INFO] Удаление тишины (порог: {top_db}дБ)")
        non_silent_intervals = librosa.effects.split(audio, top_db=top_db)
        if len(non_silent_intervals) == 0:
            print("[ПРЕДУПРЕЖДЕНИЕ] Не найдены не-тихие интервалы, возвращаем исходное аудио")
            return audio
        processed_audio = np.concatenate([audio[start:end] for start, end in non_silent_intervals])
        return processed_audio
    
    def apply_highpass_filter(self, audio: np.ndarray, cutoff: int = None) -> np.ndarray:
        """
        Применение фильтра высоких частот
        
        Args:
            audio (np.ndarray): Аудиоданные
            cutoff (int): Частота среза в Гц
            
        Returns:
            np.ndarray: Отфильтрованное аудио
        """
        if cutoff is None:
            cutoff = AUDIO_PREPROCESSOR['highpass_filter']['cutoff']
            
        print(f"[INFO] Применение фильтра высоких частот (срез: {cutoff}Гц)")
        nyquist = 0.5 * self.target_sr
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)
        return signal.filtfilt(b, a, audio)
    
    def reduce_noise(self, audio: np.ndarray, noise_clip: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Подавление шума
        
        Args:
            audio (np.ndarray): Аудиоданные
            noise_clip (np.ndarray, optional): Образец шума
            
        Returns:
            np.ndarray: Аудио с подавленным шумом
        """
        print("[INFO] Подавление шума")
        # Если образец шума не предоставлен, используем первые 1000 мс
        if noise_clip is None and len(audio) > int(self.target_sr):
            noise_clip = audio[:int(self.target_sr)]
        
        # Подавление шума с помощью библиотеки noisereduce
        try:
            nr_strength = AUDIO_PREPROCESSOR['noise_reduction']['nr_strength']
            if noise_clip is not None:
                return nr.reduce_noise(y=audio, y_noise=noise_clip, sr=self.target_sr, prop_decrease=nr_strength)
            else:
                return nr.reduce_noise(y=audio, sr=self.target_sr, prop_decrease=nr_strength)
        except Exception as e:
            print(f"[ОШИБКА] Не удалось подавить шум: {e}")
            return audio
    
    def apply_echo_reduction(self, audio: np.ndarray) -> np.ndarray:
        """
        Уменьшение эха в записях онлайн-встреч (Zoom, Google Meet)
        
        Args:
            audio (np.ndarray): Аудиоданные
            
        Returns:
            np.ndarray: Аудио с уменьшенным эхом
        """
        print("[INFO] Применение алгоритма уменьшения эха")
        try:
            # Применяем FFT для перехода в частотную область
            fft_size = 2048
            hop_length = 512
            
            # Получаем спектрограмму
            stft = librosa.stft(audio, n_fft=fft_size, hop_length=hop_length)
            magnitude, phase = librosa.magphase(stft)
            
            # Подавляем эхо путем уменьшения амплитуды в определенных частотных диапазонах
            # Типичные частоты эха в онлайн-конференциях: 300-3000 Гц
            freq_bins = librosa.fft_frequencies(sr=self.target_sr, n_fft=fft_size)
            echo_mask = np.ones_like(magnitude)
            
            # Находим индексы частотных бинов, соответствующие диапазону эха
            echo_range = (300, 3000)
            echo_indices = np.where((freq_bins >= echo_range[0]) & (freq_bins <= echo_range[1]))[0]
            
            # Применяем маску подавления к этим частотам
            echo_mask[echo_indices, :] = 0.7  # Уменьшаем на 30%
            
            # Применяем маску
            magnitude_processed = magnitude * echo_mask
            
            # Восстанавливаем сигнал
            stft_processed = magnitude_processed * phase
            audio_processed = librosa.istft(stft_processed, hop_length=hop_length)
            
            # Обрезаем до исходной длины
            if len(audio_processed) > len(audio):
                audio_processed = audio_processed[:len(audio)]
            elif len(audio_processed) < len(audio):
                audio_processed = np.pad(audio_processed, (0, len(audio) - len(audio_processed)))
            
            return audio_processed
        except Exception as e:
            print(f"[ОШИБКА] Не удалось применить уменьшение эха: {e}")
            return audio
    
    def enhance_voice(self, audio: np.ndarray) -> np.ndarray:
        """
        Улучшение разборчивости речи для записей онлайн-встреч
        
        Args:
            audio (np.ndarray): Аудиоданные
            
        Returns:
            np.ndarray: Аудио с улучшенной разборчивостью речи
        """
        print("[INFO] Улучшение разборчивости речи")
        try:
            # Применяем эквализацию для улучшения разборчивости речи
            # Усиливаем частоты в диапазоне 1000-4000 Гц (диапазон разборчивости речи)
            fft_size = 2048
            hop_length = 512
            
            # Получаем спектрограмму
            stft = librosa.stft(audio, n_fft=fft_size, hop_length=hop_length)
            magnitude, phase = librosa.magphase(stft)
            
            # Создаем маску усиления для частот разборчивости речи
            freq_bins = librosa.fft_frequencies(sr=self.target_sr, n_fft=fft_size)
            voice_mask = np.ones_like(magnitude)
            
            # Находим индексы частотных бинов для диапазона разборчивости речи
            voice_range = (1000, 4000)
            voice_indices = np.where((freq_bins >= voice_range[0]) & (freq_bins <= voice_range[1]))[0]
            
            # Усиливаем эти частоты
            voice_mask[voice_indices, :] = 1.3  # Усиливаем на 30%
            
            # Применяем маску
            magnitude_processed = magnitude * voice_mask
            
            # Восстанавливаем сигнал
            stft_processed = magnitude_processed * phase
            audio_processed = librosa.istft(stft_processed, hop_length=hop_length)
            
            # Обрезаем до исходной длины
            if len(audio_processed) > len(audio):
                audio_processed = audio_processed[:len(audio)]
            elif len(audio_processed) < len(audio):
                audio_processed = np.pad(audio_processed, (0, len(audio) - len(audio_processed)))
            
            return audio_processed
        except Exception as e:
            print(f"[ОШИБКА] Не удалось улучшить разборчивость речи: {e}")
            return audio
    
    def apply_zoom_meet_optimizations(self, audio: np.ndarray) -> np.ndarray:
        """
        Применение специальных оптимизаций для записей Zoom и Google Meet
        
        Args:
            audio (np.ndarray): Аудиоданные
            
        Returns:
            np.ndarray: Оптимизированное аудио
        """
        print("[INFO] Применение оптимизаций для Zoom/Google Meet")
        
        # Проверяем, включены ли оптимизации в конфиге
        if not AUDIO_PREPROCESSOR.get('zoom_meet_optimization', {}).get('enabled', False):
            return audio
        
        # Применяем оптимизации в определенном порядке
        if AUDIO_PREPROCESSOR.get('zoom_meet_optimization', {}).get('background_noise_suppression', False):
            # Усиленное подавление фонового шума для онлайн-встреч
            audio = self.reduce_noise(audio, noise_clip=None)
        
        if AUDIO_PREPROCESSOR.get('zoom_meet_optimization', {}).get('echo_reduction', False):
            # Уменьшение эха
            audio = self.apply_echo_reduction(audio)
        
        if AUDIO_PREPROCESSOR.get('zoom_meet_optimization', {}).get('voice_enhancement', False):
            # Улучшение разборчивости речи
            audio = self.enhance_voice(audio)
        
        return audio
    
    def trim_long_silences(self, audio: np.ndarray, frame_length: int = 1024, 
                          hop_length: int = 256, threshold: float = 0.05) -> np.ndarray:
        """
        Обрезка длинных участков тишины
        
        Args:
            audio (np.ndarray): Аудиоданные
            frame_length (int): Длина фрейма для анализа
            hop_length (int): Шаг между фреймами
            threshold (float): Порог энергии для определения тишины
            
        Returns:
            np.ndarray: Аудио с обрезанными длинными участками тишины
        """
        print("[INFO] Обрезка длинных участков тишины")
        # Вычисление энергии сигнала
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Определение фреймов с энергией выше порога
        speech_frames = energy > threshold
        
        # Преобразование индексов фреймов в индексы сэмплов
        speech_samples = np.repeat(speech_frames, hop_length)
        
        # Обрезка до длины аудио
        if len(speech_samples) > len(audio):
            speech_samples = speech_samples[:len(audio)]
        elif len(speech_samples) < len(audio):
            speech_samples = np.pad(speech_samples, (0, len(audio) - len(speech_samples)))
        
        # Возвращаем только участки с речью
        return audio[speech_samples]
    
    def visualize_audio(self, audio: np.ndarray, sr: int, title: str = "Аудиосигнал") -> None:
        """
        Визуализация аудиосигнала
        
        Args:
            audio (np.ndarray): Аудиоданные
            sr (int): Частота дискретизации
            title (str): Заголовок графика
        """
        plt.figure(figsize=(12, 4))
        librosa.display.waveshow(audio, sr=sr)
        plt.title(title)
        plt.xlabel("Время (с)")
        plt.ylabel("Амплитуда")
        plt.show()
    
    def process_audio(self, file_path: str, output_path: str = None, 
                     remove_silence_flag: bool = None, highpass_filter: bool = None,
                     noise_reduction: bool = None, normalize: bool = None,
                     visualize: bool = None) -> str:
        """
        Комплексная обработка аудиофайла
        
        Args:
            file_path (str): Путь к аудиофайлу
            output_path (str, optional): Путь для сохранения обработанного аудио
            remove_silence_flag (bool, optional): Удалять ли тишину
            highpass_filter (bool, optional): Применять ли фильтр высоких частот
            noise_reduction (bool, optional): Применять ли подавление шума
            normalize (bool, optional): Применять ли нормализацию
            visualize (bool, optional): Визуализировать ли аудио
            
        Returns:
            str: Путь к обработанному аудиофайлу
        """
        # Устанавливаем значения по умолчанию из конфига, если не указаны
        if remove_silence_flag is None:
            remove_silence_flag = AUDIO_PREPROCESSOR['silence_removal']['enabled']
        if highpass_filter is None:
            highpass_filter = AUDIO_PREPROCESSOR['highpass_filter']['enabled']
        if noise_reduction is None:
            noise_reduction = AUDIO_PREPROCESSOR['noise_reduction']['enabled']
        if normalize is None:
            normalize = AUDIO_PREPROCESSOR['normalization']['enabled']
        if visualize is None:
            visualize = AUDIO_PREPROCESSOR['visualization']['enabled']
            
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = f"{base_name}_processed.wav"
        
        # Загрузка аудио
        audio, sr = self.load_audio(file_path)
        if audio is None or sr is None:
            raise ValueError(f"[ОШИБКА] Не удалось загрузить аудиофайл: {file_path}")
        
        # Визуализация исходного аудио
        if visualize:
            self.visualize_audio(audio, sr, "Исходное аудио")
        
        # Преобразование в моно, если необходимо
        if self.mono:
            audio = self.convert_to_mono(audio)
        
        # Ресемплинг, если необходимо
        audio = self.resample(audio, sr)
        sr = self.target_sr
        
        # Применение фильтра высоких частот
        if highpass_filter:
            audio = self.apply_highpass_filter(audio)
        
        # Подавление шума
        if noise_reduction:
            audio = self.reduce_noise(audio)
        
        # Применение специальных оптимизаций для Zoom/Google Meet
        audio = self.apply_zoom_meet_optimizations(audio)
        
        # Удаление тишины
        if remove_silence_flag:
            audio = self.remove_silence(audio)
        
        # Нормализация
        if normalize:
            audio = self.normalize_audio(audio)
        
        # Сохранение обработанного аудио
        try:
            sf.write(output_path, audio, sr)
            print(f"[INFO] Сохранено обработанное аудио: {output_path}")
            
            # Визуализация обработанного аудио
            if visualize:
                self.visualize_audio(audio, sr, "Обработанное аудио")
            
            return output_path
        except Exception as e:
            print(f"[ОШИБКА] Не удалось сохранить аудио: {e}")
            if os.path.exists(output_path):
                os.remove(output_path)
            raise
