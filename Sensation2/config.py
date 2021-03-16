from dataclasses import dataclass
import math
import pyaudio

@dataclass
class config:
    """
    This is config class.
    This class is provided to avoid circle importing. 
    You can describe here any settings you want to use for all programs.
    """
    frame_rate:int = 16000
    sample_bit:int = 16
    sample_width:int = int(sample_bit/8)
    sample_range:int = 2**(sample_bit-1)
    sample_second:float = 1.0
    sample_length:int = math.floor(frame_rate*sample_second)
    channels:int = 1
    n_fft:int = 1024
    hop_length:int = round(n_fft/4)
    pyaudio_format = pyaudio.paInt16
    audio_dtype = 'int16'
    input_amplification:int = 2
    CHUNK:int = round(sample_length/input_amplification)

    # Encoder settings
    futures= 513
    length = 63