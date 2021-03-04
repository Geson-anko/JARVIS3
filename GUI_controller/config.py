from dataclasses import dataclass
import os
from os import path
from os.path import join as pathjoin

@dataclass
class config:
    current_directory:str = os.path.dirname(os.path.abspath(__file__))

    root_geometry='480x320'
    controler_title:str = 'J.A.R.V.I.S. controler'

    src_folder:str = pathjoin(current_directory,'src')
    on_image:str = pathjoin(src_folder,"on.png")
    off_image:str = pathjoin(src_folder,'off.png')
    switch_font:tuple = ("Helvetica",10)

    scroll_title:str = 'Process Switch'
    scroll_font:tuple = ("Helvetica",15)

    shutdown_title:str = 'Shutdown'
    shutdown_font:tuple = ('Helvetica',20)
    shutdown_ask:str = 'Are you sure want to shutdown ?'

if __name__ == '__main__':
    print(config.on_image)