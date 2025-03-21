import os
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
import noisereduce as nr
import matplotlib.pyplot as plt

class AudioPreprocessor:
    def __init__(self, target_sr=16000, mono=True):
        """
        Инициализация препроцессора аудио
        
        Args:
            target_sr (int): Целевая частота дискретизации (по умолчанию 16кГц)
            mono (bool): Преобразовывать ли в моно (по умолчанию True)
        """
        self.target_sr = target_sr
        self.mono = mono
    
    def load_audio(self, file_path):
        """
        Загрузка аудиофайла
        
        Args:
            file_path (str): Путь к аудиофайлу
            
        Returns:
            tuple: аудиоданные и частота дискретизации
        """
        print(f"Загрузка аудио: {file_path}")
        try:
            audio, sr = librosa.load(file_path, sr=None, mono=self.mono)
            return audio, sr
        except Exception as e:
            print(f"Ошибка при загрузке аудио: {e}")
            return None, None
    
    def resample(self, audio, orig_sr):
        """
        Ресемплинг аудио до целевой частоты дискретизации
        
        Args:
            audio (np.ndarray): Аудиоданные
            orig_sr (int): Исходная частота дискретизации
            
        Returns:
            np.ndarray: Ресемплированное аудио
        """
        if orig_sr != self.target_sr:
            print(f"Ресемплинг с {orig_sr}Гц до {self.target_sr}Гц")
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=self.target_sr)
        return audio
    
    def convert_to_mono(self, audio):
        """
        Преобразование стерео в моно
        
        Args:
            audio (np.ndarray): Аудиоданные
            
        Returns:
            np.ndarray: Моно аудио
        """
        if len(audio.shape) > 1 and audio.shape[0] == 2:
            print("Преобразование стерео в моно")
            return np.mean(audio, axis=0)
        return audio
    
    def normalize_audio(self, audio):
        """
        Нормализация амплитуды аудио
        
        Args:
            audio (np.ndarray): Аудиоданные
            
        Returns:
            np.ndarray: Нормализованное аудио
        """
        print("Нормализация амплитуды")
        max_amplitude = np.max(np.abs(audio))
        if max_amplitude > 0:
            return audio / max_amplitude
        return audio
    
    def remove_silence(self, audio, top_db=20):
        """
        Удаление тишины из аудио
        
        Args:
            audio (np.ndarray): Аудиоданные
            top_db (int): Порог в дБ для определения тишины
            
        Returns:
            np.ndarray: Аудио без тишины
        """
        print(f"Удаление тишины (порог: {top_db}дБ)")
        non_silent_intervals = librosa.effects.split(audio, top_db=top_db)
        processed_audio = np.concatenate([audio[start:end] for start, end in non_silent_intervals])
        return processed_audio
    
    def apply_highpass_filter(self, audio, cutoff=80):
        """
        Применение фильтра высоких частот
        
        Args:
            audio (np.ndarray): Аудиоданные
            cutoff (int): Частота среза в Гц
            
        Returns:
            np.ndarray: Отфильтрованное аудио
        """
        print(f"Применение фильтра высоких частот (срез: {cutoff}Гц)")
        nyquist = 0.5 * self.target_sr
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)
        return signal.filtfilt(b, a, audio)
    
    def reduce_noise(self, audio, noise_clip=None):
        """
        Подавление шума
        
        Args:
            audio (np.ndarray): Аудиоданные
            noise_clip (np.ndarray, optional): Образец шума
            
        Returns:
            np.ndarray: Аудио с подавленным шумом
        """
        print("Подавление шума")
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
            print(f"Ошибка при подавлении шума: {e}")
            return audio
    
    def trim_long_silences(self, audio, frame_length=1024, hop_length=256, threshold=0.05):
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
        print("Обрезка длинных участков тишины")
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
    
    def visualize_audio(self, audio, sr, title="Аудиосигнал"):
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
    
    def process(self, file_path, output_path=None, visualize=False, noise_reduce=True, 
                highpass=True, remove_silence_flag=True, normalize=True):
        """
        Полный процесс предобработки аудио
        
        Args:
            file_path (str): Путь к входному аудиофайлу
            output_path (str, optional): Путь для сохранения обработанного аудио
            visualize (bool): Визуализировать ли аудио до и после обработки
            noise_reduce (bool): Применять ли шумоподавление
            highpass (bool): Применять ли фильтр высоких частот
            remove_silence_flag (bool): Удалять ли тишину
            normalize (bool): Нормализовать ли амплитуду
            
        Returns:
            np.ndarray: Обработанное аудио
        """
        # Загрузка аудио
        audio, sr = self.load_audio(file_path)
        if audio is None:
            return None
        
        # Визуализация исходного аудио
        if visualize:
            self.visualize_audio(audio, sr, "Исходное аудио")
        
        # Преобразование в моно
        audio = self.convert_to_mono(audio)
        
        # Ресемплинг
        audio = self.resample(audio, sr)
        
        # Применение фильтра высоких частот
        if highpass:
            audio = self.apply_highpass_filter(audio)
        
        # Нормализация
        if normalize:
            audio = self.normalize_audio(audio)
        
        # Подавление шума
        if noise_reduce:
            audio = self.reduce_noise(audio)
        
        # Удаление тишины
        if remove_silence_flag:
            audio = self.remove_silence(audio)
        
        # Визуализация обработанного аудио
        if visualize:
            self.visualize_audio(audio, self.target_sr, "Обработанное аудио")
        
        # Сохранение обработанного аудио
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            sf.write(output_path, audio, self.target_sr)
            print(f"Обработанное аудио сохранено: {output_path}")
        
        return audio

# Пример использования
if __name__ == "__main__":
    # Создание экземпляра препроцессора
    preprocessor = AudioPreprocessor(target_sr=16000, mono=True)
    
    # Путь к входному аудиофайлу
    input_file = "path/to/input/audio.wav"
    
    # Путь для сохранения обработанного аудио
    output_file = "path/to/output/processed_audio.wav"
    
    # Обработка аудио с визуализацией
    processed_audio = preprocessor.process(
        file_path=input_file,
        output_path=output_file,
        visualize=True,
        noise_reduce=True,
        highpass=True,
        remove_silence_flag=True,
        normalize=True
    )
    
    print("Предобработка аудио завершена!")



def segment_audio(self, audio, segment_length_ms=30000):
    """
    Разделение длинного аудио на сегменты фиксированной длины
    
    Args:
        audio (np.ndarray): Аудиоданные
        segment_length_ms (int): Длина сегмента в миллисекундах
        
    Returns:
        list: Список сегментов аудио
    """
    print(f"Разделение аудио на сегменты по {segment_length_ms} мс")
    segment_length_samples = int(self.target_sr * segment_length_ms / 1000)
    segments = []
    
    for i in range(0, len(audio), segment_length_samples):
        segment = audio[i:i + segment_length_samples]
        # Если последний сегмент слишком короткий, дополняем его тишиной
        if len(segment) < segment_length_samples:
            segment = np.pad(segment, (0, segment_length_samples - len(segment)))
        segments.append(segment)
    
    return segments

def apply_vad(self, audio, frame_duration_ms=30, threshold=0.3):
    """
    Применение детектора голосовой активности (VAD)
    
    Args:
        audio (np.ndarray): Аудиоданные
        frame_duration_ms (int): Длительность фрейма в мс
        threshold (float): Порог для определения голосовой активности
        
    Returns:
        np.ndarray: Маска голосовой активности (1 - речь, 0 - не речь)
    """
    print("Применение детектора голосовой активности")
    frame_length = int(self.target_sr * frame_duration_ms / 1000)
    hop_length = frame_length // 2
    
    # Вычисление энергии сигнала
    energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Нормализация энергии
    energy_norm = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-6)
    
    # Применение порога
    vad_mask = energy_norm > threshold
    
    return vad_mask

def apply_spectral_subtraction(self, audio, noise_clip=None):
    """
    Применение спектрального вычитания для подавления шума
    
    Args:
        audio (np.ndarray): Аудиоданные
        noise_clip (np.ndarray, optional): Образец шума
        
    Returns:
        np.ndarray: Аудио с подавленным шумом
    """
    print("Применение спектрального вычитания")
    # Если образец шума не предоставлен, используем первые 1000 мс
    if noise_clip is None and len(audio) > int(self.target_sr):
        noise_clip = audio[:int(self.target_sr)]
    
    # Параметры STFT
    n_fft = 2048
    hop_length = 512
    
    # STFT сигнала
    stft_audio = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    
    # STFT шума
    stft_noise = librosa.stft(noise_clip, n_fft=n_fft, hop_length=hop_length)
    
    # Вычисление спектра мощности шума
    noise_power = np.mean(np.abs(stft_noise)**2, axis=1, keepdims=True)
    
    # Спектральное вычитание
    stft_denoised = stft_audio - np.sqrt(noise_power)
    stft_denoised = np.maximum(stft_denoised, 0)  # Ограничение отрицательных значений
    
    # Обратное STFT
    denoised_audio = librosa.istft(stft_denoised, hop_length=hop_length)
    
    # Обрезка до исходной длины
    if len(denoised_audio) > len(audio):
        denoised_audio = denoised_audio[:len(audio)]
    elif len(denoised_audio) < len(audio):
        denoised_audio = np.pad(denoised_audio, (0, len(audio) - len(denoised_audio)))
    
    return denoised_audio
