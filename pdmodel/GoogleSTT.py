import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from google.cloud import speech
from konlpy.tag import Okt
import jiwer
import jieba


credential_path = "/HDD3/leo/pdstt-403708-02d4bb4e8c3e.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path


def count_wer_tw(text, gt_text):

    reference_tokens = jieba.cut(gt_text)
    hypothesis_tokens = jieba.cut(text)

    reference = ' '.join(reference_tokens)
    hypothesis = ' '.join(hypothesis_tokens)

    return jiwer.wer(reference, hypothesis)


def tokenize_korean_text(text):
    tokenizer = Okt()
    # Tokenizes the text into words
    words = tokenizer.morphs(text)
    return " ".join(words)  # Join the tokens back into a string for WER calculation


def calculate_wer(reference, hypothesis):
    # Use the jiwer library to calculate WER
    wer = jiwer.wer(reference, hypothesis)
    return wer


def calculate_wer_korean(gt, p):
    # Tokenize the reference and hypothesis texts
    tokenized_reference = tokenize_korean_text(gt)
    tokenized_hypothesis = tokenize_korean_text(p)

    # Calculate WER
    error_rate_korean = calculate_wer(tokenized_reference, tokenized_hypothesis)

    return error_rate_korean


def visualize_voice_data(path, x, y, name=None):
    df = pd.read_csv(path)
    g = df.groupby("class")
    sns.violinplot(data=df, x=x, y=y)
    plt.savefig(f"../{name}.png")

    print(g["score"].mean())


def google_api_stt(file, config=None):
    # Instantiates a client
    client = speech.SpeechClient()
    if config:
        config = config
    else:
        config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                                          sample_rate_hertz=16000,
                                          language_code='zh-TW',
                                          )

    # Loads the audio into memory
    with open(file, 'rb') as audio_file:
        content = audio_file.read()
        audio = speech.RecognitionAudio(content=content)
        
    

    # Detects speech in the audio file long file 
    operation  = client.long_running_recognize(config=config, audio=audio)
    response = operation.result(timeout=90)
    
    # short
    # response = client.recognize(config=config, audio=audio)
    
    #select best score
    score_ls = []
    text_ls = []
    text_len = []
    
    for i, result in enumerate(response.results):
        alternative = result.alternatives[0]
        score_ls.append(alternative.confidence)
        text_ls.append(alternative.transcript)
        text_len.append(len(alternative.transcript))

    if text_len:
        best_i = np.argmax(np.array(text_len))
        score = score_ls[best_i]
        text = text_ls[best_i]
    else:
        score = 0
        text = ""

    return text, score
