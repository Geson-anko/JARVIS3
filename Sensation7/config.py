from dataclasses import dataclass
import pyaudio
import torch
import math

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
    recognize_length_rfft:int = math.floor(recognize_length/2)+1
    MFCC_channels:int = 32
    MFCC_p:float = 0.97
    kaburi:int = 1
    kaburi_step:int = int(recognize_length/kaburi)
    sample_length:int = int(sample_second * frame_rate)
    seq_len:int = int(sample_second/recognize_second)

    speak_second:float = 1.6
    speak_n_fft:int = recognize_length
    speak_mel_channels:int = 128
    speak_seq_len:int = int(speak_second/recognize_second)*kaburi
    speak_fps:int = 16000
    speak_length:int = int(speak_second*speak_fps)
    speak_range:int = 2**(bit_rate-1)
    speak_stft_length:int = math.floor(speak_length/recognize_length*2)+1
    # ---- end of Audios

    # HumanChecks ----
    HumanThreshold:float = 0
    check_dtype='float16'
    check_torch_dtype = torch.float16
    check_nfft:int = recognize_length
    check_hop_length:int = int(check_nfft//2)
    check_channels_rfft:int = recognize_length_rfft
    check_seq_len:int = int(sample_length/check_hop_length)+1
    check_CHUNK:int=recognize_length*3

    # --- end of HumanChecks

    # Texts -----
    vocab_size = 8192
    n_cluster:int = 96
    text_seq_len:int = 8
    word_dim:int = 64
    word_dtype = 'float16'
    use_corpus_length = 100000
    separator_name = 'SeparatorCent96_mfcc2'

    use_mem_len:int = 8
    generate_max_words:int = 64

    AutoEncoderDataName:str = 'AutoEncoderData.h5'
    AE_KeyName:str = 'data'