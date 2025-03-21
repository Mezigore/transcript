import os
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
import noisereduce as nr
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Union


class AudioPreprocessor:
    def __init__(self, target_sr: int = 16000, mono: bool = True):
        """
        Инициализация препроцессора аудио
        
        Args:
            target_sr (int): Целевая частота дискретизации (по умолчанию 16кГц)
            mono (bool): Преобразовывать ли в моно (по умолчанию True)
        """
        self.target_sr = target_sr
        self.mono = mono
    
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
    
    def remove_silence(self, audio: np.ndarray, top_db: int = 20) -> np.ndarray:
        """
        Удаление тишины из аудио
        
        Args:
            audio (np.ndarray): Аудиоданные
            top_db (int): Порог в дБ для определения тишины
            
        Returns:
            np.ndarray: Аудио без тишины
        """
        print(f"[INFO] Удаление тишины (порог: {top_db}дБ)")
        non_silent_intervals = librosa.effects.split(audio, top_db=top_db)
        if len(non_silent_intervals) == 0:
            print("[ПРЕДУПРЕЖДЕНИЕ] Не найдены не-тихие интервалы, возвращаем исходное аудио")
            return audio
        processed_audio = np.concatenate([audio[start:end] for start, end in non_silent_intervals])
        return processed_audio
    
    def apply_highpass_filter(self, audio: np.ndarray, cutoff: int = 80) -> np.ndarray:
        """
        Применение фильтра высоких частот
        
        Args:
            audio (np.ndarray): Аудиоданные
            cutoff (int): Частота среза в Гц
            
        Returns:
            np.ndarray: Отфильтрованное аудио
        """
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
            if noise_clip is not None:
                return nr.reduce_noise(y=audio, y_noise=noise_clip, sr=self.target_sr)
            else:
                return nr.reduce_noise(y=audio, sr=self.target_sr)
        except Exception as e:
            print(f"[ОШИБКА] Не удалось подавить шум: {e}")
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
                     remove_silence_flag: bool = True, highpass_filter: bool = True,
                     noise_reduction: bool = True, normalize: bool = True,
                     visualize: bool = False) -> str:
        """
        Комплексная обработка аудиофайла
        
        Args:
            file_path (str): Путь к аудиофайлу
            output_path (str, optional): Путь для сохранения обработанного аудио
            remove_silence_flag (bool): Удалять ли тишину
            highpass_filter (bool): Применять ли фильтр высоких частот
            noise_reduction (bool): Применять ли подавление шума
            normalize (bool): Нормализовать ли аудио
            visualize (bool): Визуализировать ли аудио до и после обработки
            
        Returns:
            str: Путь к обработанному аудиофайлу
        """
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
        
        # Удаление тишины
        if remove_silence_flag:
            audio = self.remove_silence(audio)
        
        # Нормализация
        if normalize:
            audio = self.normalize_audio(audio)
        
        # Визуализация обработанного аудио
        if visualize:
            self.visualize_audio(audio, sr, "Обработанное аудио")
        
        # Сохранение обработанного аудио
        print(f"[INFO] Сохранение обработанного аудио: {output_path}")
        sf.write(output_path, audio, sr)
        
        return output_path
    
    def segment_audio(self, audio: np.ndarray, segment_length_ms: int = 1000) -> List[np.ndarray]:
        """
        Разделение аудио на сегменты фиксированной длины
        
        Args:
            audio (np.ndarray): Аудиоданные
            segment_length_ms (int): Длина сегмента в миллисекундах
            
        Returns:
            List[np.ndarray]: Список сегментов аудио
        """
        # Вычисление длины сегмента в сэмплах
        segment_length = int(self.target_sr * segment_length_ms / 1000)
        
        # Разделение аудио на сегменты
        segments = []
        for i in range(0, len(audio), segment_length):
            segment = audio[i:i+segment_length]
            # Если последний сегмент короче, дополняем его тишиной
            if len(segment) < segment_length:
                segment = np.pad(segment, (0, segment_length - len(segment)))
            segments.append(segment)
        
        print(f"[INFO] Аудио разделено на {len(segments)} сегментов по {segment_length_ms} мс")
        return segments
    
    def apply_vad(self, audio: np.ndarray, frame_duration_ms: int = 30, 
                 aggressiveness: int = 3) -> np.ndarray:
        """
        Применение Voice Activity Detection (VAD)
        
        Args:
            audio (np.ndarray): Аудиоданные
            frame_duration_ms (int): Длительность фрейма в миллисекундах
            aggressiveness (int): Агрессивность VAD (0-3)
            
        Returns:
            np.ndarray: Маска VAD (1 - речь, 0 - тишина)
        """
        try:
            import webrtcvad
        except ImportError:
            print("[ПРЕДУПРЕЖДЕНИЕ] webrtcvad не установлен. Установите его с помощью 'pip install webrtcvad'")
            return np.ones(len(audio))
        
        vad = webrtcvad.Vad(aggressiveness)
        
        # Преобразование аудио в формат, подходящий для VAD
        frame_length = int(self.target_sr * frame_duration_ms / 1000)
        frames = []
        for i in range(0, len(audio), frame_length):
            frame = audio[i:i+frame_length]
            if len(frame) < frame_length:
                frame = np.pad(frame, (0, frame_length - len(frame)))
            frames.append(frame)
        
        # Применение VAD к каждому фрейму
        vad_mask = np.zeros(len(frames))
        for i, frame in enumerate(frames):
            # Преобразование в формат int16
            frame_int16 = (frame * 32767).astype(np.int16).tobytes()
            try:
                vad_mask[i] = vad.is_speech(frame_int16, self.target_sr)
            except:
                vad_mask[i] = 1  # В случае ошибки считаем, что это речь
        
        # Расширение маски до размера аудио
        vad_mask_expanded = np.repeat(vad_mask, frame_length)
        if len(vad_mask_expanded) > len(audio):
            vad_mask_expanded = vad_mask_expanded[:len(audio)]
        elif len(vad_mask_expanded) < len(audio):
            vad_mask_expanded = np.pad(vad_mask_expanded, (0, len(audio) - len(vad_mask_expanded)))
        
        print(f"[INFO] Применен VAD, обнаружено {np.sum(vad_mask)} фреймов с речью из {len(vad_mask)}")
        return vad_mask_expanded
    
    def apply_spectral_subtraction(self, audio: np.ndarray, 
                                  noise_clip: Optional[np.ndarray] = None,
                                  frame_length: int = 2048,
                                  hop_length: int = 512,
                                  alpha: float = 2.0) -> np.ndarray:
        """
        Спектральное вычитание для подавления шума
        
        Returns:
            np.ndarray: Аудио с подавленным шумом
        """
        print("[INFO] Применение спектрального вычитания")
        
        # Если образец шума не предоставлен, используем первую секунду аудио
        if noise_clip is None and len(audio) > self.target_sr:
            noise_clip = audio[:self.target_sr]
        elif noise_clip is None:
            print("[ПРЕДУПРЕЖДЕНИЕ] Не предоставлен образец шума и аудио слишком короткое")
            return audio
        
        # Параметры для STFT
        n_fft = frame_length
        
        # STFT для шума
        noise_stft = librosa.stft(noise_clip, n_fft=n_fft, hop_length=hop_length)
        noise_power = np.mean(np.abs(noise_stft)**2, axis=1)
        
        # STFT для аудио
        audio_stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        audio_power = np.abs(audio_stft)**2
        audio_phase = np.angle(audio_stft)
        
        # Спектральное вычитание
        power_diff = audio_power - alpha * noise_power[:, np.newaxis]
        power_diff = np.maximum(power_diff, 0.01 * audio_power)  # Спектральный пол
        
        # Восстановление сигнала
        audio_stft_denoised = np.sqrt(power_diff) * np.exp(1j * audio_phase)
        audio_denoised = librosa.istft(audio_stft_denoised, hop_length=hop_length, length=len(audio))
        
        return audio_denoised


# Пример использования
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        
        if os.path.exists(input_file):
            preprocessor = AudioPreprocessor(target_sr=16000, mono=True)
            output_file = preprocessor.process_audio(
                input_file,
                output_path=output_file,
                highpass_filter=True,
                noise_reduction=True,
                remove_silence_flag=True,
                normalize=True
            )
            print(f"[INFO] Обработанный файл сохранен как: {output_file}")
        else:
            print(f"[ОШИБКА] Файл {input_file} не найден. Укажите путь к существующему аудиофайлу.")
