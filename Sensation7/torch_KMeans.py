"""writing utf-8"""
import torch
import os 
import random
import math

class torch_KMeans:

    def __init__(self,distance=0):
        """
        distance    euclid -> 0,
                    cos_sim -> 1
        """
        if distance == 0:
            self.calc_dist = self.euclid
        elif distance == 1:
            self.calc_dist = self.cos_sim

    def KMeans(self,n_cluster:int, data:torch.Tensor, max_iter=300,centroids_init='k-means++',default_centroids=None,seed=0) -> torch.Tensor:
        """
        n_cluster : n
        data : (N,E)
        max_iter : default 300
        centroids_init : default k-means++, another option is random
        """
        assert n_cluster <= data.size(0)
        self.reset_seed(seed)
        device = data.device
        if default_centroids is None:
            if centroids_init == 'k-means++':
                centroids = self.init_centroids_pp(n_cluster,data)
            elif centroids_init == 'random':
                datamax = torch.max(torch.abs(data))
                centroids = torch.randn(n_cluster,data.size(1)).to(device)
                _cmax = torch.max(torch.abs(centroids))
                centroids /= datamax / _cmax  
            else:
                raise Exception('unknown centroids initialization method')
        else:
            centroids = default_centroids

        for _ in range(max_iter):
            classes = self.clustering(centroids,data)
            _cent = self.get_next_centroid(n_cluster,data,classes)
            if torch.sum(centroids-_cent) != 0:
                centroids = _cent.clone()
            else:
                print('\nfinished!')
                break
            print('\rProgress : {:3.1f}%'.format((_+1)/max_iter*100),end='')
        return centroids



    def init_centroids_pp(self,n_cluster: int,data: torch.Tensor) -> torch.Tensor:
        """
        n_cluster : int (n)
        data : (N,E)
        return -> (n,E)
        """
        datalen = data.size(0)
        assert n_cluster <= datalen

        centroids = torch.zeros(n_cluster+1,data.size(1),dtype=data.dtype).to(data.device)
        _first = torch.randint(0,datalen,(1,))[0]
        centroids[0] = data[_first]
        dist = torch.sum((data-data[_first].repeat(datalen,1))**2,dim=1)
        _second = torch.multinomial(dist,1)[0]
        centroids[1] = data[_second]

        for i in range(2,n_cluster):
            centroids = self.pp_next_centroids(centroids,data,now_len=i)
            print('\rgetting centroids {:3.1f}%'.format((i+1)/n_cluster * 100),end='')
        print('\ngot centroids!')
        return centroids[:n_cluster]

    def pp_next_centroids(self,centroids:torch.Tensor, data:torch.Tensor,now_len=None) -> torch.Tensor:
        """
        centroids : (n,E)
        data : (N,E)
        return -> if now_len is None, (n+1,E), else (n,E)
        """
        assert centroids.size(0) < data.size(0) and centroids.size(1) == data.size(1)
        ellen = data.size(1)
        if now_len is None:
            now_len = centroids.size(0)
            _c = torch.zeros(now_len+1,centroids.size(1)).to(centroids.device)
            _c[:now_len] = centroids
            centroids = _c

        datalen = data.size(0)

        cl,dist = self.clustering(centroids[:now_len],data,return_distances=True)
        idxes = torch.arange(datalen)
        keep_points = torch.zeros(now_len,ellen,dtype=centroids.dtype)
        keep_sumdis = torch.zeros(now_len,dtype=centroids.dtype)
        for t in range(now_len):
            b = cl==t
            use = dist[t][b]
            idx = idxes[b]
            if use.size(0) == 1 and use[0] == 0.0:
                keep_sumdis[t] = math.inf
                continue
            try:
                use_idx = torch.multinomial(use,1)[0]
            except RuntimeError:
                print(use)
                print(True in torch.isnan(use))
                print(True in torch.isinf(use))
                print(True in use < 0)
                raise Exception('Error')
            using_data = data[idx[use_idx]]
            centroids[now_len] = using_data
            keep_points[t] = using_data
            _cl,_dist = self.clustering(centroids[:now_len+1],data,return_distances=True)
            _gokei = torch.zeros(_dist.size(0))
            for d in range(_dist.size(0)):
                _gokei[d] = torch.sum(_dist[d][_cl==d])
            keep_sumdis[t] = torch.sum(_gokei)

        centroids[now_len] = keep_points[torch.argmin(keep_sumdis)]
        return centroids


    def get_next_centroid(self,n_cluster: int, data: torch.Tensor,cluster_list: torch.Tensor) -> torch.Tensor:
        """
        n_cluster : n
        data : (N,E)
        cluster_list : (N,)
        return : (n,E)
        """
        assert data.size(0) == cluster_list.size(0)
        next_c = torch.zeros(n_cluster,data.size(1)).to(data.device)
        for i in torch.unique(cluster_list):
            placevec = data[cluster_list == i]
            next_c[i] = torch.sum(placevec,dim=0) / placevec.size(0)
        del data,placevec,cluster_list
        return next_c
            
            


    def clustering(self,centroids: torch.Tensor,data: torch.Tensor,return_distances= False,full=False) -> torch.Tensor:
        """
        data : (N,E)
        centoroids : (C,E) <- C <= N
        return : [0,1,3,1 ....] <- torch.tensor
        return dist : (N,C)
        """

        datalen = data.size(0)
        centlen = centroids.size(0)
        assert data.size(-1) == centroids.size(-1)
        
        if full:
            copcent = centroids.repeat(datalen,1,1)
            _data = data.view(datalen,1,-1).repeat(1,centlen,1)
            distances = self.calc_dist(_data,copcent)
            #print(distances.size())
            cluster = torch.argmin(distances,dim=-1)
            distances = distances.T
        else:
            cluster = torch.zeros(datalen).type(torch.long).to(data.device)
            #distances = [torch.sum((data-i.repeat(datalen,1))**2,dim=1) for i in centroids]
            distances = [self.calc_dist(data,i.repeat(datalen,1)) for i in centroids]
            distances = torch.stack(distances)
            #print(distances.size())
            for t,i in enumerate(distances):
                _d = i.repeat(centlen,1)
                b = _d <= distances
                boolean = torch.prod(b,dim=0).type(torch.bool)
                cluster[boolean] = t
            del datalen,centlen,boolean,_d
        if return_distances:
            return cluster,distances
        else:
            del distances
            return cluster

    def euclid(self,data1: torch.Tensor ,data2: torch.Tensor) -> torch.Tensor:
        """
        data1 : (*N,E)
        data2 : (*N,E)
        return : (*N,)
        """
        return torch.sum((torch.sub(data1,data2))**2,dim=-1)

    def cos_sim(self,data1: torch.Tensor, data2: torch.Tensor) -> torch.Tensor:
        """
        data1 : (N,E)
        data2 : (N,E)
        return : (N,)
        cos_sim * -1 + 1
        """
        abs_ab = torch.sqrt(torch.sum(data1**2,dim=-1)) * torch.sqrt(torch.sum(data2**2,dim=-1))
        ab = torch.sum(data1*data2,dim=-1)
        cos = -torch.div(ab,abs_ab) + 1.00001
        return cos


    def reset_seed(self,seed=0):
        os.environ['PYTHONHASHSEED'] = '0'
        random.seed(seed)
        torch.manual_seed(seed)

    def predict(self,centroids: torch.Tensor,data:torch.Tensor,batch_size=16,device=None) -> torch.Tensor:
        """
        centroids : (n,E)
        data : (N,E)
        batch_size : default 16
        return -> torch.tensor([1,2,4,0,1,3...]) (N,)
        """
        if device is None:
            device = data.device
        maxlen = (data.size(0)-1) // batch_size + 1
        classes = []
        for i in range(0,data.size(0),batch_size):
            _d = data[i:i+batch_size].to(device)
            _cls = self.clustering(centroids,_d)
            classes.append(_cls)
        return torch.cat(classes).type(torch.long)


if __name__ == '__main__':
    import time 
    KMeans = torch_KMeans(1)
    dumc = torch.randn(3,10)
    dumd = torch.randn(100,10)
    cent = KMeans.get_next_centroid(3,dumd,cl)
    KMeans.KMeans(10,dumd)
    #print(torch.max(KMeans.calc_dist(dumc,dumc)))
    #print(cent)