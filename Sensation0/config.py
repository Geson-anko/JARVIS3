from dataclasses import dataclass
import os
from os.path import join as pathjoin
@dataclass
class config:
    frame_rate:int = 30
    width:int = 640
    height:int = 360
    channels:int = 3
    frame_size:tuple = (width,height)

    current_directory:str = os.path.dirname(os.path.abspath(__file__))

    encoder_params:str = pathjoin(current_directory,'params/encoder.params')
    deltatime_params:str = pathjoin(current_directory,'params/deltatime.params')
    decoder_params:str = pathjoin(current_directory,'params/decoder.params')