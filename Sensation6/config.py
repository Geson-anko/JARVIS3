from dataclasses import dataclass
import pyaudio
import torch

@dataclass
class config:
    """
    This is config class.
    This class is provided to avoid circle importing. 
    You can describe here any settings you want to use for all programs.
    """
    # Audios ---
    frame_rate:int =  16000
    channels:int = 1
    bit_rate:int = 16
    audio_dtype:str = 'int16'
    pyaudio_format = pyaudio.paInt16
    sample_width:int = int(bit_rate/8)
    sample_range:int = 2**(bit_rate-1)
    recognize_second:float = 0.04
    sample_second:float = 0.64
    recognize_length:int = int(recognize_second*frame_rate)
    sample_length:int = int(sample_second * frame_rate)
    seq_len:int = int(sample_second/recognize_second)

    speak_second:float = 1.6
    speak_seq_len:int = int(speak_second/recognize_second)
    speak_fps:int = 8000
    speak_length:int = int(speak_second*speak_fps)
    speak_range:int = 2**(bit_rate-1)
    # ---- end of Audios

    # HumanChecks ----
    HumanThreshold:float = 8.0
    # --- end of HumanChecks

    # Texts -----
    vocab_size = 8192
    n_cluster:int = 64
    text_seq_len:int = 8
    word_dim:int = 64
    word_dtype = 'float16'
    use_corpus_length = 100000
    separator_name = 'SeparatorCent64'


    use_mem_len:int = 8
    generate_max_words:int = 64
