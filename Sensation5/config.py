from dataclasses import dataclass

@dataclass
class config:
    """
    This is config class.
    This class is provided to avoid circle importing. 
    You can describe here any settings you want to use for all programs.
    """
    frame_rate:int = 30
    frame_width:int = 640
    frame_height:int = 360
    frame_size:tuple = (frame_width,frame_height)
    window_width:int = 32
    window_height:int = 32
    window_size:tuple = (window_width,window_height)
    channels:int = 3
    
    kyorokyoro:int = 1
