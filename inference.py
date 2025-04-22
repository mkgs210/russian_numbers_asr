import os
import argparse
import torch
import torchaudio
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from torchaudio.transforms import Resample, MelSpectrogram, AmplitudeToDB
from num2words import num2words

# --------------------------
# Параметры для препроцессинга
# --------------------------
TARGET_SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80

# --------------------------
# Функции декодирования и измерения расстояния
# --------------------------
# Словарь для букв
vocab = " абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
vocab_to_idx = {ch: idx + 1 for idx, ch in enumerate(vocab)}
idx_to_vocab = {idx + 1: ch for idx, ch in enumerate(vocab)}
num_classes = len(vocab) + 1  # 0 – blank

def decode_ctc(indices):
    """Декодирование последовательности индексов в строку (CTC)"""
    decoded = []
    prev = None
    for idx in indices:
        if idx != prev and idx != 0:
            decoded.append(idx)
        prev = idx
    return "".join([idx_to_vocab[i] for i in decoded if i in idx_to_vocab])

def levenshtein_distance(ref, hyp):
    """Вычисление расстояния Левенштейна между двумя строками"""
    m = len(ref)
    n = len(hyp)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1,
                           dp[i][j-1] + 1,
                           dp[i-1][j-1] + cost)
    return dp[m][n]

def words_to_number_fuzzy(predicted, candidate_min=0, candidate_max=10000):
    """
    По строке predicted (например, "двадцать один") перебираем кандидаты из диапазона candidate_min..candidate_max,
    для каждого вычисляем расстояние Левенштейна между predicted и строковым представлением числа, полученным через num2words.
    Возвращается число с минимальным расстоянием.
    """
    best_num = None
    best_distance = float("inf")
    for num in range(candidate_min, candidate_max + 1):
        candidate_str = num2words(num, lang='ru')
        # Сравниваем строки в нижнем регистре, убираем лишние пробелы
        candidate_str = candidate_str.lower().strip()
        score = levenshtein_distance(predicted, candidate_str)
        if score < best_distance:
            best_distance = score
            best_num = num
    return best_num

# --------------------------
# Определение модели (частично повторяет структуру обучения)
# --------------------------
from torchaudio.models import Conformer

class ASRLightningConformer(pl.LightningModule):
    def __init__(self,
                 input_dim=80,
                 num_classes=num_classes,
                 num_heads=4,
                 ffn_dim=1024,
                 num_layers=8,
                 depthwise_conv_kernel_size=31,
                 dropout=0.15,
                 learning_rate=0.01,
                 weight_decay=1e-4,
                 subsampling_factor=4):
        super(ASRLightningConformer, self).__init__()
        self.save_hyperparameters()
        self.conformer = Conformer(
            input_dim=input_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            use_group_norm=False,
            convolution_first=False
        )
        self.fc = nn.Linear(input_dim, num_classes)
        self.subsampling_factor = subsampling_factor

    def forward(self, x, lengths):
        def checkpointed_conformer(inp):
            out, _ = self.conformer(inp, lengths)
            return out
        # Используем gradient checkpointing для экономии памяти
        out = torch.utils.checkpoint.checkpoint(checkpointed_conformer, x, use_reentrant=False)
        out = self.fc(out)
        # Транспонируем для совместимости с декодером: [time, batch, num_classes]
        return out.transpose(0, 1)

# --------------------------
# Функция препроцессинга аудио
# --------------------------
def preprocess_audio(audio_path):
    # Загрузка аудио
    waveform, sample_rate = torchaudio.load(audio_path)
    # Если каналов несколько, берём первый
    if waveform.shape[0] > 1:
        waveform = waveform[0:1, :]
    # Если частота дискретизации не TARGET_SAMPLE_RATE, ресемплим
    if sample_rate != TARGET_SAMPLE_RATE:
        resampler = Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)
        waveform = resampler(waveform)
    # Вычисляем мел-спектрограмму
    mel_transform = MelSpectrogram(
        sample_rate=TARGET_SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    amplitude_to_db = AmplitudeToDB()
    mel_spec = mel_transform(waveform)
    mel_spec = amplitude_to_db(mel_spec)
    # Нормализация
    mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-5)
    # Как в collate_fn: удаляем размерность канала и транспонируем, чтобы получить [time, n_mels]
    spec = mel_spec.squeeze(0).transpose(0, 1)
    # Добавляем размерность батча: [1, time, n_mels]
    spec = spec.unsqueeze(0)
    # Создаём длину последовательности (число временных шагов)
    spec_lengths = torch.tensor([spec.shape[1]], dtype=torch.long)
    return spec, spec_lengths

# --------------------------
# Функция инференса
# --------------------------
def inference(audio_path, checkpoint_path, candidate_min=0, candidate_max=10000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Загружаем модель из чекпойнта
    model = ASRLightningConformer.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()
    
    # Препроцессинг аудио
    spec, spec_lengths = preprocess_audio(audio_path)
    spec = spec.to(device)
    spec_lengths = spec_lengths.to(device)
    
    with torch.no_grad():
        # Получаем выход модели
        output = model(spec, spec_lengths)
        # Если модель возвращает выход с временем, усечённым (субсемплированным), этот фактор был subsampling_factor.
        # Для декодирования CTC достаточно вывести argmax по классовой оси
        output_cpu = output.detach().cpu()
        # output_cpu: [batch, time_subsampled, num_classes] – выбираем argmax по последней оси
        preds = torch.argmax(output_cpu, dim=2)
        # Т.к. у нас батч из 1, берём первый элемент и преобразуем в список
        pred_indices = preds[0].tolist()
    
    # Декодируем последовательность символов в строку
    transcription = decode_ctc(pred_indices)
    print("Распознанный текст:", transcription)
    
    # Если результат пустой, можно задать значение по умолчанию (например, 0)
    if not transcription.strip():
        print("Пустой результат распознавания. Выводим 0.")
        return 0

    # С помощью фуззи-подбора подбираем число, строковое представление которого максимально похоже к transcription
    recognized_number = words_to_number_fuzzy(transcription, candidate_min, candidate_max)
    print("Полученное число:", recognized_number)
    return recognized_number

# --------------------------
# Главная функция для запуска инференса
# --------------------------
if __name__ == "__main__":
    # Указываем параметры вручную
    audio_path = "/path/to/audio/file.wav"  # Замените на путь к вашему аудиофайлу
    checkpoint_path = "/path/to/checkpoint.ckpt"  # Замените на путь к вашему чекпойнту
    candidate_min = 0  # Минимальное значение кандидата
    candidate_max = 999999  # Максимальное значение кандидата

    # Запускаем инференс
    inference(audio_path, checkpoint_path, candidate_min, candidate_max)
