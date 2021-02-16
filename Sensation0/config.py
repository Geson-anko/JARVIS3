from dataclasses import dataclass

@dataclass
class config:
    frame_rate:int = 30
    width:int = 640
    height:int = 360
    channels:int = 3
    frame_size:tuple = (width,height)
    