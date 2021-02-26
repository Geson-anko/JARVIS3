from dataclasses import dataclass
import math

@dataclass
class config:
    one_day:float = 24#*60*60

    sleep_time:float = 0#*60*60
    wake_time:float = 6#*60*60
    before_sleep:float = 3#*60*60
    after_wake:float = 2#*60*60

    threshold:float = 0.999
    My:float= 0.5

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