import os
import sys
from typing import Tuple
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

from MemoryManager import MemoryManager
from MasterConfig import Config as mconf
from .config import config

import pyaudio
import numpy as np
import multiprocessing as mp
from datetime import datetime,timedelta
import time

class SleepManager(MemoryManager):
    LogTitle:str = 'SleepManager'
    def __init__(self,debug_mode: bool=False) -> None:
        super().__init__(self.LogTitle, debug_mode=debug_mode)
        if not os.path.isdir(config.temp_folder):
            os.mkdir(config.temp_folder)
            self.log(f'made {config.temp_folder}')
        
        # calculate sleep curve function ------------------
        _w = config.wake_time
        if _w <= config.sleep_time:
            _w += config.one_day
        sfs_len = _w - config.sleep_time
        wfs_len = config.one_day - sfs_len
        self.wake_switch = (config.sleep_time + sfs_len/2)%config.one_day 
        self.sleep_switch = (config.wake_time + wfs_len/2)%config.one_day 
        
        _s = config.sleep_time
        if _s <= self.sleep_switch:
            _s += config.one_day
        _hb = _s - config.before_sleep/2
        T,M = np.log(1/config.threshold - 1),np.log(1/config.Meany - 1)
        self.sleep_k = (T - M) / (_s - _hb)
        self.sleep_b = T - self.sleep_k * _s

        _w = config.wake_time
        if _w <= self.wake_switch:
            _w += config.one_day
        _hb = _w + config.after_wake/2
        self.wake_k = (T-M)/(_w - _hb)
        self.wake_b = T - self.wake_k * _w
        # -----------------------------------------------------
    
    def activation(
        self,shutdown:mp.Value,sleep:mp.Value,switch:mp.Value,clock:mp.Value,sleepiness:mp.Value,
        newest_ids:Tuple[mp.Value]
        ) -> None:
        """
        shutdown:    multiprocessing shared memory bool value.
        sleep:    multiprocessing shared memory bool value.
        switch: multiprocessing shared memory bool value.
        clock:  multiprocessing shared memory dubble value.
        sleepiness:  multiprocessing shared memory dubble value.
        newest_ids: Tuple of mutiprocessing shared memory int value.
        """    
        # load from file
        if os.path.isfile(config.DeltaBioClock_file):
            delta_bio_clock = self.load_python_obj(config.DeltaBioClock_file)
            self.log(f'loaded {config.DeltaBioClock_file}')
        else:
            delta_bio_clock = timedelta()
        
        # set audio streamer
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=config.pyaudio_format,
            channels=config.audio_channels,
            rate=config.audio_fps,
            frames_per_buffer=config.CHUNK,
            input=True,
        )
        self.log('setted audio streaming')
    
        # set default values
        Warn = False
        warntime = 0.0
        old_ids = [i.value for i in newest_ids]
        self.log('setted default values',debug_only=True)

        ## process ------------------------------------------
        self.log('process start')
        while not shutdown.value:
            # clock log
            clock_start = time.time()
            if not switch.value:
                self.log('switch off.')
                while not switch.value and not shutdown.value:
                    clock.value = 0.0
                    time.sleep(mconf.wait_time)
                self.log('switch on.')
                
            # warning check
            sound = stream.read(config.CHUNK)
            sound = np.frombuffer(sound,config.audio_dtype).reshape(-1,config.audio_channels)
            if np.max(sound) > config.warning_threshold:
                Warn = True
                self.log('turn on warning mode')
                warntime = time.time()
            
            ## sleepiness
            news = [i.value for i in newest_ids]
            modified = self.check_id_modified(news,old_ids)
            old_ids = news
            now = datetime.now(mconf.TimeZone) + delta_bio_clock
            now_clock = now.hour * 60*60 + now.minute*60 + now.second
            if not Warn:
                sleepiness.value = self.SleepinessCurve(now_clock)
                if sleepiness.value > config.threshold:
                    if not modified:
                        sleep.value = True
                    else:
                        delta_bio_clock -= config.clock_warp
                elif sleepiness.value > config.Meany and sleepiness.value <= config.threshold and not modified:
                    delta_bio_clock += config.clock_warp
                elif sleepiness.value > config.Meany and sleepiness.value <= config.threshold and modified:
                    delta_bio_clock -= config.clock_warp
                else:
                    sleep.value = False
            else:
                sleepiness.value = 0
                sleep.value = False
            
            if time.time() - warntime > config.warning_delay:
                Warn = False
            time.sleep(config.system_sleep)
            clock.value = time.time() - clock_start
        # ----------------------------------------------------
        ## shutdown process
        self.log('shutdown process started')
        self.save_python_obj(config.DeltaBioClock_file,delta_bio_clock)


    def check_id_modified(self,newids:Tuple[int],oldids:Tuple[int]) -> bool:
        modified = False
        for (p,q) in zip(newids,oldids):
            if p != q:
                modified = True
                break
        return modified


    def sleep_sigmoid(self,x:float) -> float:
        if self.sleep_switch > x:
            x += config.one_day
        v = 1/(1 + np.exp(self.sleep_k * x + self.sleep_b))
        return v
    
    def wake_sigmoid(self,x:float) -> float:

        if self.wake_switch > x:
            x += config.one_day
        v = 1 / (1+np.exp(self.wake_k*x+ self.wake_b))
        return v
    
    def SleepinessCurve(self,x:float) -> float:
        x = x % config.one_day
        if x < self.wake_switch or x >= self.sleep_switch:
            v = self.sleep_sigmoid(x)
        else:
            v = self.wake_sigmoid(x)
        return v