
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from skimage.feature import peak_local_max
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import path
import COM
# sample 1
# bragg_peaks=np.array([[ 54,  56],
#                     [ 33,  65],
#                     [ 36,  43],
#                     [ 15,  52],
#                     [ 30,  87],
#                     [ 73,  69],
#                     [ 38,  21],
#                     [ 12,  73],
#                     [ 76,  48],
#                     [ 17,  30],
#                     [ 70,  91],
#                     [ 94,  61],
#                     [ 49, 100],
#                     [ 51,  78],
#                     [ 79,  26],
#                     [ 57,  35],
#                     [ 60,  13],
#                     [ 20,   8],
#                     [ 97,  40],
#                     [ 92,  83],
#                     [ 9, 96],
#                     [ 28, 109],
#                     [42,0],
#                     [81,  4],
#                     [100,  17],
#                     [ 89, 105]
#                     ])
# num_line=[5,5,5,6,5]
# remove_index=frozenset((9,14,15,16,17,24))
# sample 2
# bragg_peaks=np.array([[ 55,  55],
#        [ 59,  26],
#        [ 73, 101],
#        [ 86,  15],
#        [ 32,  37],
#        [ 36,   8],
#        [ 78,  73],
#        [ 51,  84],
#        [ 28,  66],
#        [ 24,  95],
#        [ 82,  44],
#        [  9,  19],
#        [101,  91],
#        [106, 62],
#        [110,35],
#        [5,49],
#        [2,78]       ])
# num_line=[3,4,3,4,3]
# remove_index=frozenset((5,8,9,10,11,14))

# sample 3
bragg_peaks=[]
num_line=[3,3,3,3,3]
remove_index=frozenset((3,6,7,10))
k0=[66,65]
k1=[69,95]
k2=[42,81]
i_range=np.array([[-3,2],[-2,2],[-1,2],[0,2],
                    [-2,1],[-1,1],[0,1],[1,1],
                    [-2,0],[-1,0],[0,0],[1,0],[2,0],
                    [-1,-1],[0,-1],[1,-1],[2,-1],
                    [-1,-2],[0,-2],[1,-2],[2,-2],[3,-2]])
# crop in k space
crop=np.full((128,128),True)
crop[:,85:]=False
class q_stat:
    def __init__(self,T_idx,x,y,workingdir,auto):
        '''
        workingdir: str, datainput
        '''
        self.logdata,self.T=self._read(T_idx,x,y,workingdir)
        self.mask_ind=self._find_peak()
        self.bragg_peaks=self._generate_bragg_peaks(auto=auto)

    
    def _read(self,T_idx,x,y,workingdir):
        f=h5py.File(workingdir+'data.hdf5','r')
        I,T=f['I'],f['T']
        raw_data=I[T_idx,x,y]
        raw_data[raw_data<0]=1
        return np.log10(raw_data),T[T_idx]

    def _find_peak(self):
        mask_ind=peak_local_max(self.logdata,threshold_rel=0,min_distance=3,p_norm=2)
        return mask_ind   
    
    def _generate_bragg_peaks(self,auto):
        if auto:
            com=COM.COM(data=self.logdata,crop=crop,k0=k0,k1=k1,k2=k2,i_range=i_range)
            return com.generate_com()
        else:
            return bragg_peaks
            

    def all_peaks(self,remove_outlier=True, return_outlier=False,threshold=2):
        dist_nn=self._get_neighbors(self.mask_ind)
        if remove_outlier:
            dist_nn, cdw_peaks, bragg_peaks=self._remove_outlier(dist_nn,threshold)
            if return_outlier:
                fig,ax=plt.subplots(figsize=(4,4))
                ax.pcolormesh(self.logdata,cmap='gray')
                ax.scatter(cdw_peaks[:,1]+0.5,cdw_peaks[:,0]+0.5,s=5,color='r')
                ax.scatter(bragg_peaks[:,1]+0.5,bragg_peaks[:,0]+0.5,s=5,color='b')
                ax.set_title('{} C'.format(self.T))

        return self._get_metrics(dist_nn)
    
    def remove_bragg(self,remove_outlier=False):
        intensity=self.logdata[tuple(self.mask_ind.T)]
        labels=self._get_bragg_peaks(intensity)
        dist_nn=self._get_neighbors(self.mask_ind[labels==1])
        if remove_outlier:
            dist_nn,_,_=self._remove_outlier(dist_nn)
        
        return self._get_metrics(dist_nn)        
        
    
    def _get_bragg_peaks(self,intensity):
        '''
        return : label 0 for bragg peaks, 1 for CDW peaks
        '''
        X=intensity.reshape((-1,1))
        X=StandardScaler().fit_transform(X)
        clusters=KMeans(2,init=np.array([[X.max()],[.25*X.max()+.75*X.min()]]),n_init=1,tol=1e-2)
        clusters.fit(X)
        tmp=[intensity[clusters.labels_==label].mean() for label in range(2)]
        return clusters.labels_ if tmp[0]>tmp[1] else 1-clusters.labels_
    
    def _remove_outlier(self,dist_nn,threshold=2):
        dist,dist_occ=np.unique(dist_nn,return_counts=True)
        idx=-1
        while dist_occ[idx]<=threshold:
            idx-=1
        outlier_mask=(dist_nn>dist[idx]).flatten()
        return dist_nn[~outlier_mask], self.mask_ind[~outlier_mask],self.mask_ind[outlier_mask]
        
    def _get_neighbors(self,idx):
        nn=NearestNeighbors(n_neighbors=2)
        nn.fit(idx)
        dist_nn,_=nn.kneighbors(n_neighbors=1)
        return dist_nn

    def _get_metrics(self,dist_nn):
        return {'mean':dist_nn.mean(),'std':dist_nn.std(),'length':dist_nn.shape[0],'bragg':self.bragg_peaks}     

    def _generate_triangle(self,remove_central_shell=False,):
        bragg_peaks=self.bragg_peaks
        bragg_peaks_sort_idx=np.argsort(bragg_peaks[:,0])[::-1]
        brag_sort=np.array([bragg_peaks[bragg_peaks_sort_idx,0],bragg_peaks[bragg_peaks_sort_idx,1]]).T
        mask_ind_dict={}
        i,j,count,ptr=0,0,0,0
       
        num_line_cumsum=np.cumsum(num_line)
        for idx,value in enumerate(brag_sort):
            mask_ind_dict[(i,j)]=value
            count+=1
            if count>=num_line_cumsum[ptr]:
                    i+=1
                    j=0
                    ptr+=1
            else:
                    j+=1        
        
        bragg_triangle_list=[]
        for i in range(5):
            for j in range(num_line[i]-1):
                if i>0:
                    up= j+1 if i%2==0 else j
                    bragg_triangle_list.append([mask_ind_dict[(i,j)],mask_ind_dict[(i,j+1)],mask_ind_dict[(i-1,up)]])

                if i<4:
                        down= j+1 if i%2==0 else j
                        bragg_triangle_list.append([mask_ind_dict[(i,j)],mask_ind_dict[(i,j+1)],mask_ind_dict[(i+1,down)]])

        if remove_central_shell:
            bragg_triangle_list=[val for idx, val in enumerate(bragg_triangle_list) if idx not in remove_index]
        return bragg_triangle_list

    def _centroid(self):
        return [(tr[0]+tr[1]+tr[2])/3 for tr in  self.bragg_triangle_list]

    def visualize_grid(self,remove_central_shell=False):
        self.bragg_triangle_list=self._generate_triangle(remove_central_shell)
        fig,ax=plt.subplots(figsize=(4,4))
        ax.pcolormesh(self.logdata,cmap='gray')
        ax.set_title('{} C'.format(self.T))
        centroid=self._centroid()
        for triangle in self.bragg_triangle_list:
                pth=np.array(triangle+[triangle[0]])
                ax.plot(pth[:,1]+.5,pth[:,0]+.5,color='blue',lw=2,ls='dotted')
        for idx, cor in enumerate( centroid):
            ax.text(cor[1],cor[0],'{:.0f}'.format(idx),color='cyan',fontsize=8)  


    def count_pts(self,visualization=False,repel=1,remove_central_shell=False):
        self.bragg_triangle_list=self._generate_triangle(remove_central_shell)
        inside_num,contains_pts=_is_inside(self.bragg_triangle_list,self.mask_ind,repel=repel)
        if visualization:
            centroid=self._centroid()
            fig,ax=plt.subplots(figsize=(4,4))
            ax.pcolormesh(self.logdata,cmap='gray')
            ax.scatter(self.mask_ind[:,1]+0.5,self.mask_ind[:,0]+0.5,s=5,color='r')
            for triangle in self.bragg_triangle_list:
                pth=np.array(triangle+[triangle[0]])
                ax.plot(pth[:,1]+.5,pth[:,0]+.5,color='blue',lw=2,ls='dotted')
            for idx, cor in zip(inside_num, centroid):
                ax.text(cor[1],cor[0],'{:.0f}'.format(idx),color='cyan',fontsize=8)        
            ax.set_title('{} C'.format(self.T))

        return self._get_metrics(inside_num)

def _is_inside(tri_pts,test_pts,repel=0):
    contains=np.zeros(len(tri_pts))
    contains_pts=[]
    for idx,tri_pt in enumerate(tri_pts):
            p=path.Path(tri_pt)
            is_inside=p.contains_points(test_pts)
            test_inside=test_pts[is_inside]
            test_inside_repel=[pt for pt in test_inside if (np.linalg.norm(pt-np.array(tri_pt),axis=1).min()>repel)]
            contains[idx]=len(test_inside_repel)
            contains_pts.append(test_inside_repel)
    return contains,contains_pts
