from dataclasses import dataclass
import math
import os
from datetime import timedelta
import pyaudio

@dataclass
class config:
    one_day:float = 24*60*60 # second

    sleep_time:float = 0*60*60 # second
    wake_time:float = 5*60*60 # second
    before_sleep:float = 3*60*60 # second
    after_wake:float = 2*60*60 # second

    threshold:float = 0.999
    Meany:float= 0.5

    clock_warp:timedelta = timedelta(seconds=7*60)

    wait_time:float = 0.1
    system_sleep:float = 0.01
    # files
    current_directory:str = os.path.dirname(os.path.abspath(__file__))
    temp_folder:str = os.path.join(current_directory,'temp')
    DeltaBioClock_file:str = os.path.join(temp_folder,'delta_bio_clock.pkl')

    # audio streaming
    pyaudio_format:int = pyaudio.paInt16
    audio_channels:int = 1
    audio_fps:int = 8000
    CHUNK:int = int(round(wait_time*audio_fps))
    audio_dtype:str = 'int16'
    warning_threshold:float = 2**(16 - 1) * 0.9
    warning_delay:float = 10*60 # second

if __name__ == '__main__':
    _w = config.wake_time
    if _w <= config.sleep_time:
        _w += config.one_day
    sfs_len = _w - config.sleep_time
    wfs_len = config.one_day - sfs_len
    wake_switch = (config.sleep_time + sfs_len/2)%config.one_day
    sleep_switch = (config.wake_time + wfs_len/2)%config.one_day
    print(wake_switch,sleep_switch)

    _s = config.sleep_time
    if _s <= sleep_switch:
        _s += config.one_day
    _hb = _s - config.before_sleep/2
    print(_hb)

    _w = config.wake_time
    if _w <= wake_switch:
        _w += config.one_day
    _hb = _w + config.after_wake/2
    print(_hb)