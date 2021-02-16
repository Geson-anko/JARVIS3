from dataclasses import dataclass
import os

@dataclass
class config:
    frame_rate:int = 30
    width:int = 640
    height:int = 360
    channels:int = 3
    frame_size:tuple = (width,height)

    current_directory:str = os.path.dirname(os.path.abspath(__file__))
