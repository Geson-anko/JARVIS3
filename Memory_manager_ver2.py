import numpy as np
import torch
import os
import random
import h5py
import warnings

class Memory_manager:
    '''
    このプログラムでは、
    10進数→62進数　(int2string)
    62進数→10進数　(string2int)
    ID → 10進数
    ファイルの保存、ファイルの読み出し、存在確認、削除
    IDからフォーマットナンバーとIDから数字を出します
    忘却曲線からboolを作ります
    '''
    def __init__(self,char=None):
        if char is None:
            self.char = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
            self.base = len(self.char)
        else:
            self.char = char
        self._torchtype = type(torch.randn(1))
        self._nptype = type(np.random.randn(1))
        self._listtype = type([])
        self.fileform = '{0}/{1}.h5'

    def encode(self,num):
        if type(num) is type(''):
            num = int(num)
        if num < 62:
            return self.char[num]

        string = ''
        while True:

            string = self.char[num%self.base] + string
            num = num // self.base
            if num ==0:
                break
        return string

    def decode(self,string):
        if type(string) is type(0):
            string = str(string)
        if len(string) == 1:
            return self.char.index(string)
        num = 0
        for ch in string:
            num = num * self.base + self.char.index(ch)
        return num

    def save_file(self,ID_list,memdata,mem_head_folder,return_same_ID_warning=False):
        '''
        ID_list : List. 
        memdata : numpy  ndarray or torch tensor
        mem_head_folder : 記憶フォルダの最上位フォルダ type string
        '''
        if len(ID_list) == 0 or len(memdata) == 0:
            return None
        if len(ID_list) != len(memdata):
            raise Exception('save_file error! Not match length.')
        datatype = type(memdata)
        if self._torchtype is datatype:
            memdata = memdata.to('cpu').detach().numpy()
        memdata = memdata.astype(np.float16)
        #行列化
        memdata = memdata.reshape(len(memdata),-1)
        maxlen = len(ID_list)
        print('Saving memories to folder...\n')
        name = self.fileform.format(mem_head_folder,ID_list[0][0])
        ap = self.getabspath(name)

        with h5py.File(ap,mode='a',swmr=True) as f:
            for t,(ID,data) in enumerate(zip(ID_list,memdata)):
                if ID in f:
                    if return_same_ID_warning:
                        warnings.warn(f'existing id ! {ID}')
                else:
                    f.create_dataset(name=ID,data=data)
                per = str((t+1) / maxlen*100)[:4]
                print(f'\rProgress : {per}',end='')                    


    def exist_check(self,ID_list,mem_head_folder):
        '''
        ID_list : List. 
        mem_head_folder : 記憶フォルダの最上位フォルダ type string
        '''
        if ID_list == []:
            return []
        name= self.fileform.format(mem_head_folder,ID_list[0][0])
        ap = self.getabspath(name)
        with h5py.File(ap,mode='r') as f:
            exist_id = [i for i in ID_list if i in f]
            
        return exist_id
    
    def remove_file(self,ID_list,mem_head_folder):
        if ID_list == []:
            return None
        name = self.fileform.format(mem_head_folder,ID_list[0][0])
        ap = self.getabspath(name)
        print('removing file:\n',ID_list)
        with h5py.File(ap,mode='a') as f:
            for i in ID_list:
                if i in f:
                    del f[i]        

    def load_file(self,ID_list,mem_head_folder,gpu=False,usenumpy=False,return_id=True):
        '''
        ID_list : List. 
        mem_head_folder : 記憶フォルダの最上位フォルダ type string
        gpu : torch.tensorをGPUに送るか否か。GPUがない場合、False
        usenumpy : numpy 配列として読みだすか(Trueの場合,gpu=Falseになる)
        return_id : 読み出したIDのリストを返すかどうか
        '''
        if ID_list == []:
            empty = np.array([],dtype='float16')
            if gpu and usenumpy:
                import warnings
                warnings.warn('You tryed use gpu by numpy, but can not use.')
                return [],empty
            elif gpu and usenumpy == False:
                if torch.cuda.is_available():
                    empty = torch.from_numpy(empty).to('cuda')
                else:
                    import warnings
                    warnings.warn('You tryed use gpu, but can not use.')   
                return [],empty
            elif usenumpy and gpu == False:
                return [],empty
            else:
                return [],torch.from_numpy(empty)

        name = self.fileform.format(mem_head_folder,ID_list[0][0])
        ap = self.getabspath(name)
        exist_id = []
        exap = exist_id.append
        data = []
        daap = data.append
        with h5py.File(ap,mode='r') as f:
            for ID in ID_list:
                if ID in f:
                    exap(ID)
                    daap(f[ID][:])
        data = np.array(data,dtype=np.float16)
        if usenumpy:
            gpu = False
        else:
            data = torch.from_numpy(data)

        if gpu:
            if torch.cuda.is_available():
                data = data.to('cuda')
            else :
                import warnings
                warnings.warn('You tryed use gpu, but can not use.')                
        
        if return_id :
            return exist_id,data
        else:
            return data
    
    def ID2num(self,ID):
        '''
        return : idform,idnum
        '''
        num_zone = ID[1:]
        format_zone = ID[0]
        idnum = self.decode(num_zone)
        idform = self.decode(format_zone)
        return idform,idnum

    def get_nextID(self,id,a=1):
        """
        ミスってエンコードとデコード逆に書いちゃった
        """
        form,num = id[0],id[1:]
        enum = self.decode(num)
        enum +=a
        dnum = self.encode(enum)

        numlen = len(num)
        dnum = '0'*(numlen-len(dnum))+dnum

        if numlen < len(dnum):
            raise Exception('Memory ID was over flow!')
        else:
            return form+dnum
        
    def make_forgetbool(self,max_mems):
        """
        max_mems : int
        """
        x = np.array([*range(max_mems)])
        memprob = np.exp(-5/max_mems*x)
        forgprob = 1-memprob
        tf = [True,False]
        calc_prob = memprob < 0.9
        cmps,cfps = memprob[calc_prob],forgprob[calc_prob]
        forg_bool = [False] * (memprob.shape[0] - cmps.shape[0])
        for i in zip(cfps,cmps):
            forg_bool += random.choices(tf,weights=i)
        return np.array(forg_bool)

    def forget(self,ID_ranking,mem_head_folder,forget_bool=None,max_mems=-1,news_first=True):
        """
        ID_ranking : ID list
        forget_bool: numpy bool array, (True: forget)
        max_mems : int, len(forget_bool) = max_mems
        news_first: True , [new, old] (default True)
        """

        if news_first:
            typerank = type(ID_ranking)
            if typerank is self._listtype:
                ID_ranking.reverse()
            elif typerank is self._nptype:
                ID_ranking = ID_ranking[::-1]
            else:
                raise Exception('Unknow list type entered!')
        ID_ranking = np.array(ID_ranking)
        fgbtype = type(forget_bool)
        if fgbtype is self._listtype:
            forget_bool = np.array(forget_bool)
        
        if forget_bool is None:
            import warnings
            warnings.warn('Please make forget bool before forget process.')
            if max_mems >0:
                forget_bool = self.make_forgetbool(max_mems)
            else:
                raise Exception('please set only forget_bool or max_mems')
        
        ranklen = ID_ranking.shape[0]
        if max_mems == -1:
            max_mems= forget_bool.shape[0]
        if ranklen > max_mems:
            forg0,ID_ranking = ID_ranking[:-max_mems],ID_ranking[-max_mems:]
        else:
            forg0 = np.array([]).astype(ID_ranking.dtype)
        
        ranklen = ID_ranking.shape[0]
        forget_bool = forget_bool[:ranklen][::-1]
        switch = forget_bool == False
        forg1,ID_ranking = ID_ranking[forget_bool],ID_ranking[switch]
        forg = forg0.tolist() + forg1.tolist()
        if forg == []:
            self.remove_file(forg,mem_head_folder)
        if news_first:
            ID_ranking = ID_ranking[::-1]
        return ID_ranking.tolist()

    @staticmethod
    def getabspath(file_name):
        dirs = os.path.dirname(os.path.abspath(__file__)).split('\\')
        dirs = '/'.join(dirs) +'/'+ file_name
        return dirs

    @staticmethod
    def same_ID_format(ID_list,format_head=None):
        """
        ID_list : list
        format_head: None or string, if None. ID_list[0][0]
        """
        if ID_list == []:
            return []
        if format_head is None:
            format_head = ID_list[0][0]
        ID_new = [i for i in ID_list if i[0] == format_head]
        return ID_new

    def get_itemsize(self,x):
        itemtype = type(x)
        if itemtype is self._nptype:
            itms = x.itemsize
            nums = x.shape
        elif itemtype is self._torchtype:
            itms = x.element_size()
            nums = x.size()
        else:
            raise Exception('unknow array or tensor!')
        for i in nums:
            itms =  itms * i
        return itms
    
    @staticmethod
    def version():
        print('version2.2')