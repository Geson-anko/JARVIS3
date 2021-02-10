import pandas as pd
import numpy as np
import torch
import itertools
flatter = itertools.chain.from_iterable
import random
import copy
import gc

class searching_memories:
    def __init__(self,VRAM_limit='2G'):
        self._torchtype = type(torch.randn(1))
        self._nptype = type(np.random.randn(1))
        self._inttype = type(1)
        self._strtype = type('')
        self.vramlim = self.get_VRAM_limit(VRAM_limit)
        self._dataframetype = type(pd.DataFrame([]))
    
    def get_VRAM_limit(self,VRAM_limit):
        if type(VRAM_limit) == self._strtype:
            per = VRAM_limit[-1].upper()
            rnum = int(VRAM_limit[:-1])
        else:
            per = ''
        if per == 'G':
            ramlim = rnum * 10**9
        elif per == 'M':
            ramlim =  rnum * 10**6
        elif per == 'K':
            ramlim = rnum * 10**3
        else:
            ramlim = int(VRAM_limit)
        return ramlim


    def get_itemsize(self,x):
        itemtype = type(x)
        if itemtype == self._nptype:
            itms = x.itemsize
            nums = x.shape
        elif itemtype == self._torchtype:
            itms = x.element_size()
            nums = x.size()
        else:
            raise Exception('unknow array or tensor!')
        for i in nums:
            itms =  itms * i
        return itms
    
    def search(self,mem,one_time_mems,one_time_ids,brunch,depth,*mem_dicts,Trans_mem_dict=None,VRAM_limit=None):
        '''
        mem: encoded data. shape is (E,), dtype=torch.float16, torch.CudaTensor
        one_time_mems: shape is (N,E), dtype=torch.float16,
        one_time_ids: shape is (N,), type list
        brunch: type int
        depth: type int
        *mem_dicts: memory dictionaries.(The form is {A:[B,C]})
        Trans_mem_dict: pandas DataFrame format, matching_mem_dict
        VRAM_limit: limit of VRAM. string or int
        '''
        if one_time_ids == []: # is empty
            return [],np.array([],dtype='float16')

        # type match check
        memtype = type(mem)
        otmtype = type(one_time_mems)
        if memtype != otmtype:
            raise Exception(f'Not match tensor type!\nmem:{memtype},one_time_mem:{otmtype}')
        #calc VRAM <-- 2000000000 Byte limit (default)
        if VRAM_limit !=None:
            print('sorry, I was not able to  make VRAM saver....')
            self.vramlim = self.get_VRAM_limit(VRAM_limit)
#        otmsize = self.get_itemsize(one_time_mems)
#        if otmsize * 2 < self.vramlim:
#            batchify = False
#        else:
#            batchify = True
        # cuda check
        if str(one_time_mems.device) == 'cpu':
            one_time_mems = one_time_mems.to('cuda')
        if mem.device != one_time_mems.device:
            mem = mem.to(one_time_mems.device)
        
        # calculate distance
        otmlen = len(one_time_mems)
        otilen = len(one_time_ids)
        if otmlen != otilen:
            raise Exception('one time mems length and one time ids length is not matching')
#        memsize = self.get_itemsize(mem)
#        if batchify:
#            empty = self.vramlim - otmsize
#            if empty < 0:
#                raise Exception('one time memories has exceeded the VRAM limit')
#            batch = otmsize // empty + 1
#            for i in range(0,otmsize,batch):
#                batchlen = 
#       else:
        remem = mem.repeat(otmlen,1)
        distance = torch.sqrt(torch.sum((remem-one_time_mems)**2,1))
        topmemarg = torch.argsort(distance).to('cpu').detach().numpy()[:brunch] # <- change torch to numpy 
        #distance = distance.to('cpu')
        #topdistance = torch.min(distance)
        searched = [one_time_ids[i] for i in topmemarg]
        searched0 = copy.copy(searched)
        _depth = range(depth)
        _tmdic = Trans_mem_dict != None
        if _tmdic:
            if type(Trans_mem_dict) != self._dataframetype :
                raise Exception('If you use Trans mem dict, please set a Pandas.DataFrame.')

        for ID in searched0:
            brunched = [ID]
            for _ in _depth:
                ichiji = list(flatter([dic[bid] for dic in mem_dicts for bid in brunched if bid in dic]))
                if _tmdic:
                    ichiji += self.trans_search(brunched,Trans_mem_dict)
                brunched_num = [*range(len(ichiji))]
                random.shuffle(brunched_num)
                brunched_num = brunched_num[:brunch]
                brunched = list(set([ichiji[i] for i in brunched_num]))
                searched += brunched
        searched = list(dict.fromkeys(searched))
        del mem,one_time_mems,searched0,topmemarg,remem
#        gc.collect()
#        torch.cuda.empty_cache()
        return searched,distance#.detach().numpy()


    def trans_search(self,id_list,Trans_mem_dict):
        """
        id_list : list ex.) [id1,id2,id3,...]
        Trans_mem_dict : pandas Data Frame
        """
        columns = Trans_mem_dict.columns.values
        setcolumns = set(columns)
        matches = []
        for i in id_list:
            formatnum = i[0]
            if formatnum in setcolumns:
                matched = Trans_mem_dict[Trans_mem_dict[formatnum] == i]
                matched = list(flatter(matched.itertuples(index=False,name=None)))
                matches += matched

            else:
                pass
        return list(set(matches) - set(id_list))
