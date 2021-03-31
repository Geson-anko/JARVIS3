import os
import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
import h5py
import pickle
from os.path import isdir, join as pathjoin
from MasterConfig import Config
from debug_tools import Debug
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.shared_memory import SharedMemory
from typing import Optional,List,Tuple,Any, Union
import gc

@dataclass
class MemoryManager(Debug): 
    r"""
    This is a parent class for all processes.

    ***methods explanation***
    
    Num2Id:
        A integer translate to decimal N.
        Using characters are IDchars in Config.py

    Id2Num:
        A memory id (translated by Num2Id) translate to decimal 10.
        Using charscters are IDchars in Config.py
    
    save_memory:
        saving memories, got_time, and ID. You can enter unsigned integer or ID string with list.
        if you entered integer, this program call 'Num2Id'.

    load_memory:
        loading memories. return_time is used training delta time model only.
        You can enter unsigned integer or ID string with list.
        if you enterd integer, this program calls 'Num2Id'

    extract_sameId:
        Extract only 1 id format. You often use for output process.
            Ex.) ['01110','0asd1','10001'] -> ['01110','0asd1']

        Please be carefull Not to use torch.Tensor.

    get_firstId:
        You can use for sensation processes when first run.

    create_shared_memory:
        create multiprocessing shared memory object. 
        We only use for main.py

        ** Warning !! **
            Never forget call destory_shared_memory after you called this method!!!!!!!

    destory_shared_memory:
        destory all shared memory object in the process.

    inherit_shared_memory:
        You need to inherit shared memory object for some process

    in_ReadOutId:
        get index from ReadOutId if Id is in ReadOutId.
        ** Warning !! **
        ReadOutId must be sorted because this method uses Binary Search Algorhythm. Otherwise ...
    
    load_memory_withReadOuts:
        load memory preferentially from ReadOutMemory.
        ** Warning !! **
        This method use in_ReadOutId method. So ReadOutId must be sorted because it uses Binary Search Algorhythm. Otherwise ...

    save_python_obj:
        saving python objects to pickle file.
    
    load_python_obj:
        loading python objects from pickle file.

    remove_file:
        remove the file.
    
    release_ststem_memory:
        This method releases system memory caches. To release cache is heavy process so please don't use quite often. 

    *** Parent methods ***
    log:
        this method is used instead of print() and write to log file.
        Example:
            >>> debug = Debug('log_title',debug_mode=False)
            >>> debug.log('test') # <- Instead of print()
    
    exception:
        This method is used instead of 'raise Exception()' and write to log file.
        Example:
            >>> debug = Debug('log_title',debug_mode=False)
            >>> if before_the_error:
            >>>     debug.excepion('error!')

    warn:
        This method is used instead of 'wanings.warn' and write to log file.
        Please Be carefull that 'text' argument is not variable length.
        Example:
            >>> debug = Debug('log_title',debug_mode=False)
            >>> if danger:
            >>>     debug.warn(text)
    """
    char:str = Config.IDchars
    base:int = Config.decimal_base

    data_dir:str = '{}/data'
    time_dir:str = '{}/time'

    return_time:bool = False

    

    def __init__(self,log_title:str='',debug_mode:bool=False) -> None:
        """
        log_title is used Exceptions and UserWarnings.
        """
        super().__init__(log_title,debug_mode)
        #self.debug = Debug(log_title,debug_mode)
        self.public_shared_memory = []

        memory_folder = pathjoin(Config.current_directory,Config.memory_folder)
        if not isdir(memory_folder):
            os.makedirs(memory_folder)

        a= len(Config.IDchars) **(Config.ID_length-1)
        self.memory_format_integers = np.arange(len(Config.IDchars)+1,dtype=Config.ID_dtype)*a


    def Num2Id(self,num:int) -> str:
        """
        integer to ID. (deciam 10 -> decimal N)
        
        return -> string
        """
        
        _t = type(num)
        if _t in Config.str_types:
            self.warn(f'your input type is {_t}. There is Num2Id method to convert str ID to int ID.')
            return num
        elif not _t in Config.int_types:
            self.exception(f'your input is {_t}! Please integer type!')

        if num < 0:
            self.exception(f'your input is {num} < 0 ! Please unsigned integer !')
        string = ''
        while True:

            string = self.char[num%self.base] + string
            num = num // self.base
            if num ==0:
                break
        sl = len(string)
        if sl < Config.ID_length:
            string = self.char[0] * (Config.ID_length - sl) + string
        return string

    def Id2Num(self,string:str) -> int:
        """
        ID to integer. (decimal N -> decimal 10)
        
        return -> a integer
        """

        _t = type(string)
        if _t in Config.int_types:
            self.warn(f'your input type is {_t}. There is Id2Num method to convert int ID to str ID.')
            return string
        elif _t not in Config.str_types:
            self.exception(f'your input is {_t} !. Please {str}')
        
        if len(string) == 1:
            return self.char.index(string)
        num = 0
        for ch in string:
            num = num * self.base + self.char.index(ch)
        return num

    
    def save_memory(
        self,ID:List[str] or str or int or List[int] or ndarray,
        memory_data:ndarray or Tensor,
        memory_time:List[float] or float or ndarray) -> None:
        """
        ID_list [required]: list. Ex -> [id,...] or id 
        memory_data [required]: numpy ndarray or torch tensor
        memory_time [required]: list. The time when memory data was got.
        """
        
        _idt = type(ID)
        _mdt = type(memory_data)
        _mtt = type(memory_time)

        if _idt is ndarray:
            ID = ID.tolist()
            _idt = type(ID)
        if _mtt is ndarray:
            memory_time = memory_time.tolist()
            _mtt = type(memory_time)

        if _mdt is Tensor:
            memory_data = memory_data.to('cpu').detach().numpy().astype('float16')
            _mdt = ndarray

        # saving a memory
        if _idt is int or _idt is str:
            if _mtt is list:
                self.exception(f'You try to save a memory with list of time. Please give a memory and a time.')
            if _idt is int:
                ID = self.Num2Id(ID)
            name = Config.memory_file_form.format(ID[0])
            #name = pathjoin(Config.current_directory,Config.memory_file_form.format(ID[0]))
            with h5py.File(name,'a',swmr=True) as f:
                if ID in f:
                    self.warn(f'{ID} is existing!')
                else:
                    ddir = self.data_dir.format(ID)
                    tdir = self.time_dir.format(ID)
                    f.create_dataset(name=ddir,data=memory_data)
                    f.create_dataset(name=tdir,data=memory_time)
            return None

        # saving memories process.
        il,dl,tl = len(ID),len(memory_data),len(memory_time)
        if il != dl or tl != il:
            self.exception(f'save file error! Not match length. ID:{il},data:{dl},time:{tl}')

        if il == 0:
            return None

        elem_type = type(ID[0])
        
        if elem_type is int:
            ID = [self.Num2Id(i) for i in ID]
        name =   Config.memory_file_form.format(ID[0][0])
        #name = pathjoin(Config.current_directory,Config.memory_file_form.format(ID[0][0]))

        with h5py.File(name,'a',swmr=True) as f:
            allid = f.keys() 
            for (i,d,t) in zip(ID,memory_data,memory_time):
                if i in allid:
                    self.warn(f'{i} is existing!')
                else:
                    ddir = self.data_dir.format(i)
                    tdir = self.time_dir.format(i)
                    f.create_dataset(name=ddir,data=d)
                    f.create_dataset(name=tdir,data=t)
        
    def _exist_memory(
        self,ID:List[str] or str or int or List[int] or ndarray) -> Tuple[List[bool],List[str]]:
        """
        ID [required]: Ex -> [id,...] or id 
        return -> existing bools and existed id_str. 
                if you entered list of Id, return Tuple[List[bool],List[str]].
        """

        _idt = type(ID)
        if _idt is ndarray:
            ID = ID.tolist()
            _idt = type(ID)

        if _idt is int or _idt is str:
            if _idt is int:
                ID = self.Num2Id(ID)
            name = Config.memory_file_form.format(ID[0])
            exist = False
            with h5py.File(name,'a',swmr=True) as f:
                if ID in f:
                    exist = True
            return exist, ID
    
    def _load_a_memory(
        self,ID:str) -> Tuple[str,ndarray,float]:
        """
        this program is a load a memory.
        if you don't want to return time, please 'self.return_time = False'

        return :
            self.return_time is False -> Tuple[str,ndarray]
            self.return_time is True  -> Tuple[str,ndarray,time]
        """
        name = Config.memory_file_form.format(ID[0])
        #name = pathjoin(Config.current_directory,Config.memory_file_form.format(ID[0]))

        try:
            with h5py.File(name,'r',swmr=True) as f:
                ddir = self.data_dir.format(ID)
                data = np.array(f[ddir])
                if self.return_time:
                    tdir = self.time_dir.format(ID)
                    time = np.array(f[tdir])
                    #print(time)
                    return ID,data,time
                else:
                    return ID,data
        except KeyError:
            return None
        except OSError:
            return None
        
    def load_memory(
        self,ID:List[str] or str or int or List[int] or ndarray,
        return_time:bool = False,return_id_int:bool=False) -> Tuple[List[str or int],ndarray,ndarray]:
        """
        this program is load memories.
        ID : [str,...], [int,...], str, int or the ndarray of these are included in elements.
        
        if you entered int or str, return is Tuple[str,memory(ndarray),time(ndarray)]
        if you entered list of int or str, return is Tuple[List[id],ndarray,ndarray]
        When return_time is False, the return is without time. return Tuple[id,memory]
        """
        self.return_time = return_time

        _idt = type(ID)
        if _idt is ndarray:
            ID = ID.tolist()
            _idt = type(ID)
        
        if _idt is int or _idt is str:
            if _idt is int:
                ID = self.Num2Id(ID)
            return self._load_a_memory(ID)
        
        if _idt is not list:
            self.exception(f'you entered {_idt}, Please enter list,ndarray,int,str')
        
        if len(ID) == 0:
            if return_time:
                return [],np.array([]),np.array([])
            else:
                return [],np.array([])


        elem_type = type(ID[0])
        if elem_type is int:
            ID = [self.Num2Id(i) for i in ID]
        
        with ThreadPoolExecutor() as p:
            result = p.map(self._load_a_memory,ID)
        data,time,exist_id = [],[],[]
        dap,tap,eap = data.append,time.append,exist_id.append

        for i in result:
            if i is not None:
                eap(i[0])
                dap(i[1])
                if return_time:
                    tap(i[2])
        #print(time)
        if len(data) > 0:
            data = np.stack(data) 
            if return_time:
                time = np.stack(time)
        else:
            data = np.array(data)
            if return_time:
                time = np.array(time) 

        if return_id_int:
            exist_id = [self.Id2Num(i) for i in exist_id]
        if return_time:
            return exist_id,data,time
        else:
            return exist_id,data
    """
    def _extract_sameId(self,ID:List[str or int] or ndarray,id_format:str or int,return_integer:bool=True) -> Union[List[str],List[int],ndarray]:
        
        extracting same id formats.
        id_format [required] : extract this format from ID_list.
        

        if len(ID) == 0:
            return []
        
        _idt = type(ID)
        if _idt is not list and _idt is not ndarray:
            self.exception(f'your input type is {_idt}, please list or ndarray')
        if _idt is ndarray:
            ID = ID.tolist()

        elem_type = type(ID[0])

        
        if elem_type is int:
            ID = [self.Num2Id(i) for i in ID]
        
        _ift  = type(id_format)
        if _ift is not str and _ift is not int:
            self.exception(f'please int or str for id_format. your input is {_ift}.')

        if _ift is int:
            if id_format > self.base:
                self.exception(f' over format number. your format id number is {id_format}')
            else:
                id_format = self.char[id_format]
        
        extracted = [i for i in ID if i[0] == id_format]
        if return_integer:
            extracted = [self.Id2Num(i) for i in extracted]
            if _idt is ndarray:
                extracted = np.array(extracted,dtype=Config.ID_dtype)

        return extracted
    """
    def extract_sameId(self,ID:List[str or int] or ndarray,id_format:str or int,return_integer:bool=True) -> Union[List[str],List[int],ndarray]:
        """
        extracting same id formats.
        id_format [required] : extract this format from ID_list.
        """
        if len(ID) == 0:
            return []
        _idt = type(ID)
        _t = type(ID[0])
        str_mode = False

        if _idt is not list and _idt is not ndarray:
            self.exception(f'your input type is {_idt}, please list or ndarray.')
        if _t in Config.str_types:
            str_mode = True
        elif _t not in Config.int_types:
            self.exception(f'your input element type is {_t}. please int or str type.')

        _ift  = type(id_format)
        if str_mode:
            # type check
            if _ift in Config.int_types:
                id_format = self.char[id_format]
            elif _ift not in Config.str_types:
                self.exception(f'please int or str for id_format. your input is {_ift}.')
            
            # extracting
            ID = [i for i in ID if i[0] == id_format]

            # type convertings
            if return_integer:
                ID = [self.Id2Num(i) for i in ID]
            
        else:
            # type check
            if _ift in Config.str_types:
                id_format = self.char.index(id_format)
            elif _ift not in Config.int_types:
                self.exception(f'please int or str for id_format. your input is {_ift}.')
            if _idt is list:
                ID = np.array(ID,dtype=Config.ID_dtype)
            
            # extracting
            ID = ID[(self.memory_format_integers[id_format] <= ID) & (ID < self.memory_format_integers[id_format+1])]
            
            # type convertings
            if _idt is list:
                ID = ID.tolist()
            if not return_integer:
                ID = [self.Num2Id(i) for i in ID]

        
        return ID
            
    def get_firstId(self,id_format:str or int,return_integer=True) -> Union[int,str]:
        """
        return int when return_integer is True.
        Please use sesation process.
        """
        _ift  = type(id_format)
        if _ift is not str and _ift is not int:
            self.exception(f'please int or str for id_format. your input is {_ift}.')
        


        if _ift is int:
            if id_format > self.base:
                self.exception(f' over format number. your format id number is {id_format}')
            else:
                id_format = self.char[id_format]
        
        start_id = id_format + '0'*(Config.ID_length -1)
        if return_integer:
            start_id = self.Id2Num(start_id)
        return start_id
            
    def _create_shared_memory(
        self,shape:Tuple[int,...],
        dtype:str or np.dtype,
        initialize:Any=None) -> Tuple[ndarray,SharedMemory]:
        """
        shape [required]: tuple of int. Ex.) (10,5)
        dtype [required]: str or numpy.dtype
        initialize:[optional]: initialize specific value when it isn't None.
        """
        size = int(np.dtype(dtype).itemsize * np.prod(shape))
        pubmem = SharedMemory(create=True,size=size)
        array = ndarray(shape=shape,dtype=dtype,buffer=pubmem.buf)

        if initialize is not None:
            array[:] = initialize
        self.public_shared_memory.append(pubmem)

        return array,pubmem

    def create_shared_memory(
        self,shape:Tuple[int,...],
        dtype:str or np.dtype,
        initialize:Any=None) -> Tuple[tuple,np.dtype,SharedMemory]:
        """
        shape [required]: tuple of int. Ex.) (10,5)
        dtype [required]: str or numpy.dtype
        initialize:[optional]: initialize specific value when it isn't None.
        """
        size = int(np.dtype(dtype).itemsize * np.prod(shape))
        pubmem = SharedMemory(create=True,size=size)
        array = ndarray(shape=shape,dtype=dtype,buffer=pubmem.buf)

        if initialize is not None:
            array[:] = initialize
        self.public_shared_memory.append(pubmem)
        return array.shape,array.dtype,pubmem

    
    def inherit_shared_memory(self,created_shared_memory:Tuple[tuple,np.dtype,SharedMemory]) -> ndarray:
        """
        created_shared_memory [requierd]: Please use create_shared_memory before this method.
        """
        shape,dtype,pubmem = created_shared_memory
        return ndarray(shape=shape,dtype=dtype,buffer=pubmem.buf)

    def destory_shared_memory(self) -> Any:
        """
        destoring shared memory objects.
        """

        for i in self.public_shared_memory:
            i.close()
            i.unlink()
        self.public_shared_memory.clear()
        gc.collect()

    
    def _inherit_shared_memory(self,created_shared_memory:Tuple[ndarray,SharedMemory]) -> ndarray:
        """
        created_shared_memory [requierd]: Please use create_shared_memory before this method.
        """
        array,pubmem = created_shared_memory
        return ndarray(shape=array.shape,dtype=array.dtype,buffer=pubmem.buf)


    def in_ReadOutId(
        self,Id_num:List[int] or ndarray,
        ReadOutId:ndarray) -> Tuple[ndarray,ndarray]  :
        """
        get index from ReadOutId if Id is in ReadOutId.
        Id_num [required] : [int,...].
        ReadOutId [required] : [int,...] and must be sorted.
        """
        _it = type(Id_num)
        if _it is not list and _it is not ndarray:
            self.exception(f'your input type is {_it}, please list or ndarray')

        if _it is list:
            Id_num = np.array(Id_num)
        
        if type(ReadOutId) is not ndarray:
            self.exception(f'your input ReadOutId is {type(ReadOutId)},please {np.ndarray}')
        
        outidx = np.searchsorted(ReadOutId[ReadOutId!=Config.init_id],Id_num)
        outidx[outidx == ReadOutId.shape[0]] = -1
        outbools = ReadOutId[outidx] == Id_num
        return outidx,outbools

    def load_memory_withReadOuts(
        self,ID_num:Union[List[int],ndarray],ReadOutId:ndarray,ReadOutMemory:ndarray
        ) -> ndarray:
        """
        This method is load_memory wrapper.
        Please be carefull Not to return ID.

        Id_num [required]:  [int,...]
        ReadOutId [required]:   ndarray[int,...]
        ReadOutMemory [required]:   ndarray[float,...]
        This method will be faster than using load_memory because it gets preferentially from ReadOutMemory.
        ** Warning !! **
        This method use in_ReadOutId method, So ReadOutId must be sorted because it uses Binary Search Algorhythm. Otherwise ...
        Ex.)
            >>> useid = self.extract_sameId(TemporaryMemory,'0')
            >>> memory = self.load_memory_withReadOuts(
            ... useid,
            ... ReadOutId, # sorted 
            ... ReadOutMemory)
            >>>
        """
        _it = type(ID_num)
        if _it is not list and _it is not ndarray:
            self.exception(f'your input type is {_it}, please list or ndarray')
        if _it is list:
            ID_num = np.array(ID_num)

        outidx,outbools = self.in_ReadOutId(ID_num,ReadOutId)
        memory = ReadOutMemory[outidx[outbools]]
        loadId = ID_num[np.logical_not(outbools)] 
        if loadId.shape[0] != 0:
            ids,mem = self.load_memory(loadId,return_id_int=False,return_time=False)
            if mem.shape[0] != 0:
                memory = np.concatenate([memory,mem])
        return memory

    @staticmethod
    def save_python_obj(file_name:str,obj:Any) -> None:
        """
        saving python object to pickle file
        """
        with open(file_name,'wb') as f:
            pickle.dump(obj,f,protocol=4)

    @staticmethod
    def load_python_obj(file_name:str) -> Any:
        """
        loading python object from pickle file
        """
        with open(file_name,'rb') as f:
            obj = pickle.load(f)
        return obj
    
    def remove_file(self,file_name:str) -> None:
        """
        delete file
        """
        if os.path.exists(file_name):
            os.remove(file_name)
        else:
            self.warn(f'Not exists {file_name}')
    @staticmethod
    def release_system_memory() -> None:
        gc.collect()
        torch.cuda.empty_cache()
        

    def activation(self):pass
    def __call__(self,*args,**kwargs):
        return self.activation(*args,**kwargs)



if __name__ == '__main__':
    import time
    mm = MemoryManager()
    dummyid = np.array([0,1,2])
    dummydata = np.random.randn(3,10)
    dummytime = [0.0,1.0,2.0]

    #print(mm.load_memory(dummyid,return_time=True))
    #print(mm.extract_sameId(dummyid,''))
    #print(mm.get_firstId(60,True))
    #print(mm.Id2Num('Y000000000'))
    #print(mm.create_shared_memory((1,),'float16',1))
    #mm.destory_shared_memory()

