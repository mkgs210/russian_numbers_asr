# import pandas as pd
# import whisper
# import os
# from tqdm import tqdm
# import torch
# import torch.nn as nn
# import torchaudio
# import torchaudio.transforms as transforms
# import shutil

# def augment_audio(waveform, sample_rate, output_path):
#     # Создаем трансформации для спектрограммы
#     log_mel_spec_transform = transforms.MelSpectrogram(
#         sample_rate=sample_rate,
#         n_mels=128,
#         win_length=400,
#         hop_length=160,
#         n_fft=1024
#     )
    
#     # Создаем аугментации
#     time_masks = [torchaudio.transforms.TimeMasking(time_mask_param=15, p=0.05) for _ in range(10)]
#     spec_augment = nn.Sequential(
#         transforms.FrequencyMasking(freq_mask_param=25),
#         *time_masks
#     )
    
#     # Получаем спектрограмму
#     log_mel_spec = log_mel_spec_transform(waveform)
#     log_mel_spec = torch.log(log_mel_spec + 1e-14)
    
#     # Применяем аугментации
#     augmented_log_mel_spec = spec_augment(log_mel_spec)
    
#     # Сохраняем аугментированное аудио
#     torchaudio.save(output_path, waveform, sample_rate)

# def process_audio_files(csv_path, output_csv_path, augment=True):
#     # Загрузка модели Whisper
#     model = whisper.load_model("large-v3")
    
#     # Чтение CSV файла
#     df = pd.read_csv(csv_path)
    
#     # Добавляем новую колонку для текстовой транскрипции
#     df['text_transcription'] = ''
    
#     # Создаем папки для аугментированных данных
#     if augment:
#         augmented_folder = 'dev_augmented' if 'dev' in csv_path else 'train_augmented'
#         os.makedirs(augmented_folder, exist_ok=True)
        
#         # Создаем DataFrame для аугментированных данных
#         augmented_rows = []
    
#     # Обработка каждого аудиофайла
#     for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {csv_path}"):
#         audio_path = row['filename']
        
#         if not os.path.exists(audio_path):
#             print(f"File not found: {audio_path}")
#             continue
            
#         try:
#             # Транскрибация аудио
#             result = model.transcribe(audio_path)
#             transcription = result['text'].strip()
#             df.at[idx, 'text_transcription'] = transcription
            
#             # Аугментация данных
#             if augment:
#                 waveform, sample_rate = torchaudio.load(audio_path)
                
#                 # Создаем путь для аугментированного файла
#                 aug_filename = f"aug_{os.path.basename(audio_path)}"
#                 aug_path = os.path.join(augmented_folder, aug_filename)
                
#                 # Применяем аугментацию
#                 augment_audio(waveform, sample_rate, aug_path)
                
#                 # Добавляем информацию об аугментированном файле
#                 new_row = row.copy()
#                 new_row['filename'] = aug_path
#                 new_row['text_transcription'] = transcription
#                 augmented_rows.append(new_row)
                
#         except Exception as e:
#             print(f"Error processing {audio_path}: {str(e)}")
#             df.at[idx, 'text_transcription'] = ''
    
#     # Добавляем аугментированные данные в DataFrame
#     if augment and augmented_rows:
#         augmented_df = pd.DataFrame(augmented_rows)
#         df = pd.concat([df, augmented_df], ignore_index=True)
    
#     # Сохранение результатов в новый CSV файл
#     df.to_csv(output_csv_path, index=False)
#     print(f"Saved results to {output_csv_path}")

# def main():
#     # Обработка dev.csv
#     process_audio_files('dev.csv', 'dev_text.csv')
    
#     # Обработка train.csv
#     process_audio_files('train.csv', 'train_text.csv')

# if __name__ == "__main__":
#     main()