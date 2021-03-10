from dataclasses import dataclass

@dataclass
class config:
    """
    This is config class.
    This class is provided to avoid circle importing. 
    You can describe here any settings you want to use for all programs.
    """
    frame_rate:int = 30
    width:int = 640
    height:int  = 360
    channels:int = 3
    frame_size:tuple = (width,height)