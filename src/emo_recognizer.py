import torch
import torchaudio
from transformers import AutoTokenizer, AutoFeatureExtractor, Trainer, TrainingArguments
import os
import importlib.util
import sys
import logging

# Настраиваем логирование с меньшей детализацией для производительности
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

def load_custom_model(model_name):
    """Загрузка пользовательской модели с оптимизациями"""
    from huggingface_hub import snapshot_download
    
    # Скачиваем модель локально с кэшированием
    model_path = snapshot_download(repo_id=model_name)
    
    # Загружаем пользовательский модуль
    custom_module_path = os.path.join(model_path, "audio_text_multimodal.py")
    spec = importlib.util.spec_from_file_location("audio_text_multimodal", custom_module_path)
    custom_module = importlib.util.module_from_spec(spec)
    sys.modules["audio_text_multimodal"] = custom_module
    spec.loader.exec_module(custom_module)
    
    # Получаем классы из пользовательского модуля
    WavLMBertConfig = custom_module.WavLMBertConfig
    WavLMBertForSequenceClassification = custom_module.WavLMBertForSequenceClassification
    
    # Загружаем конфигурацию
    config = WavLMBertConfig.from_pretrained(model_path)
    
    # Загружаем модель
    model = WavLMBertForSequenceClassification.from_pretrained(
        model_path,
        config=config,
        local_files_only=True
    )
    
    # Проверяем модель
    if not hasattr(model, 'config') or not hasattr(model.config, 'id2label'):
        raise ValueError("Не удалось загрузить конфигурацию модели с метками эмоций")
    
    return model

class EmotionDataset(Dataset):
    def __init__(self, audio_text_pairs, processor, feature_extractor):
        self.pairs = audio_text_pairs
        self.processor = processor
        self.feature_extractor = feature_extractor
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        audio_path, text = self.pairs[idx]
        
        # Process audio
        speech_array, sampling_rate = torchaudio.load(audio_path)
        resampler = torchaudio.transforms.Resample(sampling_rate, self.feature_extractor.sampling_rate)
        speech = resampler(speech_array).squeeze().numpy()
        
        # Prepare inputs
        audio_inputs = self.feature_extractor(
            speech,
            sampling_rate=self.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True
        )
        
        text_inputs = self.processor(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        return {
            "input_values": audio_inputs["input_values"].squeeze(),
            "audio_attention_mask": audio_inputs["attention_mask"].squeeze().float(),
            "input_ids": text_inputs["input_ids"].squeeze(),
            "text_attention_mask": text_inputs["attention_mask"].squeeze().bool(),
            "token_type_ids": text_inputs["token_type_ids"].squeeze() if "token_type_ids" in text_inputs else None
        }

class EmotionRecognizer:
    def __init__(self, model_name="aniemore/wavlm-bert-fusion-s-emotion-russian-resd", device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = load_custom_model(model_name)
        
        if device == 'cuda':
            self.model = self.model.cuda()
            self.use_fp16 = torch.cuda.is_available() and hasattr(torch.cuda, 'amp')
        else:
            self.use_fp16 = False
            try:
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
            except Exception:
                pass
        
        self.model.eval()
    
    def batch_recognize(self, audio_text_pairs):
        # Create dataset
        dataset = EmotionDataset(audio_text_pairs, self.tokenizer, self.feature_extractor)
        
        # Configure trainer
        training_args = TrainingArguments(
            output_dir="./tmp_trainer",
            per_device_eval_batch_size=1,
            remove_unused_columns=False,
            no_cuda=(self.device != 'cuda'),
            fp16=self.use_fp16
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
        )
        
        # Get predictions
        predictions = trainer.predict(dataset)
        probs = torch.softmax(torch.tensor(predictions.predictions), dim=1)
        
        # Process results
        results = []
        available_labels = {int(k): v for k, v in self.model.config.id2label.items()}
        
        for prob in probs:
            result = {}
            for i, p in enumerate(prob.numpy()):
                if i in available_labels:
                    label = available_labels[i]
                    result[label] = float(p)
            results.append(result)
        
        return results

    def recognize(self, audio_path, text):
        """Single sample recognition"""
        results = self.batch_recognize([(audio_path, text)])
        return results[0]
