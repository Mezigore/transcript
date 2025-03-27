from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import torch
from config import DIARIZATION, API_KEYS


# Функция для диаризации с помощью Pyannote
def diarize(audio_tensor: torch.Tensor, sample_rate: int, hf_token=None):
    # Если токен не передан, используем из конфигурации
    if hf_token is None:
        hf_token = API_KEYS['huggingface']
        
    print("[INFO] Начало диаризации аудио...")
    
    # Загружаем модель диаризации один раз
    print("[INFO] Загрузка модели диаризации...")
    pipeline = Pipeline.from_pretrained(DIARIZATION['model_path'], use_auth_token=hf_token)
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    elif torch.backends.mps.is_available():
        pipeline.to(torch.device("mps"))
    else:
        pipeline.to(torch.device("cpu"))
    
    # # Диаризация каждого сегмента
    # print("[INFO] Выполнение диаризации...")
    all_speaker_segments = []
    
    
    try:
        with ProgressHook() as hook:
            # Get the raw diarization results
            diarization_results = pipeline({"waveform": audio_tensor.to(torch.float32), "sample_rate": sample_rate}, 
                                          min_speakers=DIARIZATION['min_speakers'], 
                                          max_speakers=DIARIZATION['max_speakers'],
                                          num_speakers=DIARIZATION['num_speakers'],
                                          hook=hook,
                                      )
            
            # Process the results into a list of segments
            for turn, _, speaker in diarization_results.itertracks(yield_label=True):
                all_speaker_segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })
            
    except Exception as e:
        print(f"[ОШИБКА] Не удалось выполнить диаризацию: {e}")
    
    print(f"[INFO] Найдено {len(all_speaker_segments)} сегментов с {len(set([s['speaker'] for s in all_speaker_segments]))} спикерами")
    return all_speaker_segments
