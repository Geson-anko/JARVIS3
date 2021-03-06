"""writing utf-8"""
from typing import Any
import warnings
from MasterConfig import Config
import os
from datetime import datetime

class Debug:
    r"""
    This class is a debug tool class.
    Please use these methods instead of 'print', 'raise Exception', etc...
    Methods of this class are write a log file.

    Example:
        >>> debug = Debug('log_title',debug_mode=False):
        >>> debug.log('test') # <- Instead of print()
    
    """

    def __init__(self,log_title:str,debug_mode:bool=False):
        """
        log_title [required] : title of logs
        debug_mode [Optional] : When True, methods of class, which 'debug_only' argument been True, turn on.
        """

        self._title = f'{log_title} :'
        self._debug = debug_mode
        if debug_mode:
            self._title += '<debug mode>'
        _dir = f'{Config.current_directory}/{Config.log_dir}'
        if os.path.isdir(Config.log_dir) is False:
            os.makedirs(Config.log_dir)
        self.log_file = os.path.join(Config.log_dir,f'{log_title}.txt')


    def log(self,*args:Any,debug_only:bool=False) -> None:
        """
        this method is used instead of print() and write to log file.
        Example:
            >>> debug = Debug('log_title',debug_mode=False)
            >>> debug.log('test') # <- Instead of print()
        """
        if debug_only and self._debug is False:
            return None
        print(self._title,*args)
        self.log_write(self.now(),*args)

    def exception(self,*args:Any,debug_only:bool=False) -> None:
        """
        This method is used instead of 'raise Exception()' and write to log file.
        Example:
            >>> debug = Debug('log_title',debug_mode=False)
            >>> if before_the_error:
            >>>     debug.excepion('error!')
        """
        if debug_only and self._debug is False:
            return None

        self.log_write(self.now(),'<Exception>',*args)
        raise Exception(self._title,*args)

    def warn(self,text:str,warning_category:Warning=None,debug_only:bool=False) -> None:
        """
        This method is used instead of 'wanings.warn' and write to log file.
        Please Be carefull that 'text' argument is not variable length.
        Example:
            >>> debug = Debug('log_title',debug_mode=False)
            >>> if danger:
            >>>     debug.warn(text)

        """
        if debug_only and self._debug is False:
            return None
        if warning_category is None:
            warning_category = Warning
        
        self.log_write(self.now(),str(warning_category),text)
        warnings.warn(f'{self._title}{text}',warning_category)

    @staticmethod
    def now() -> str:
        return datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')
    
    def log_write(self,*args):
        with open(self.log_file,'a',encoding='utf-8') as f:
            message = ' '.join([str(i) for i in args])
            f.write(f'{message}\n')
        



if __name__ == '__main__':
    d = Debug('test')
    d.log(1,2,3)