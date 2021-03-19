from dataclasses import dataclass
from pynput import mouse
@dataclass
class config:
    """
    This is config class.
    This class is provided to avoid circle importing. 
    You can describe here any settings you want to use for all programs.
    """
    mouse_elems:int = 12
