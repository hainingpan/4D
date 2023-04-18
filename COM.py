import numpy as np
from scipy import ndimage
from sklearn.neighbors import NearestNeighbors
class COM:
    def __init__(self,data,k0=[54,54],k1=[51,83],k2=[28,65]):
        self.data=data
        self.k0=np.array(k0)
        self.k1=np.array(k1)
        self.k2=np.array(k2)
        self.i_range=np.array([[-2,2],[-1,2],[0,2],
                               [-2,1],[-1,1],[0,1],[1,1],
                               [-1,0],[0,0],[1,0],
                               [-1,-1],[0,-1],[1,-1],[2,-1],
                               [0,-2],[1,-2],[2,-2]])
        self.line=generate_line()
    def generate_x_i(self):
        x_i=self.i_range@np.array([self.k1-self.k0,self.k2-self.k0])+self.k0
        return np.round(x_i).astype(int)
    def generate_mask(self,x_i=None,radius=3):
        if x_i is None:
            x_i=self.generate_x_i()
        mask=np.zeros(self.data.shape,dtype=int)
        x_i_all=[]
        for i in range(-radius,radius+1):
            for j in range(-radius,radius+1):
                x_i_all.append(x_i+np.array([i,j]))
        x_i_all=np.vstack(x_i_all)
        x_i_all[x_i_all<0]=0
        x_i_all[x_i_all[:,0]>=self.data.shape[0]]=self.data.shape[0]
        x_i_all[x_i_all[:,1]>=self.data.shape[1]]=self.data.shape[1]
        mask[tuple(x_i_all.T)]=1
        return mask
    def generate_com(self):
        mask=self.generate_mask()
        lbl=ndimage.label(mask)
        return np.array(ndimage.center_of_mass(self.data,lbl[0],range(1,lbl[1]+1)))

    def average_dist(self):
        com_i=com.generate_com()
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(com_i)
        distances, indices = nbrs.kneighbors(com_i)
        return distances[:,1].mean()
    def average_dist_direction(self):
        com_i=com.generate_com()
        dist={}
        ave_dist=[]
        for key,val in self.line.items():
            dist[key]=[]
            for pts_idx in val:
                pts=com_i[pts_idx]
                dist[key].append(distance(pts))
        # return dist
        for key,val in dist.items():
            ave_distance=np.hstack(dist[key]).mean()
            ave_dist.append(ave_distance)
        return np.array(ave_dist)


def generate_line():
    line={}
    line[0]=[[0,1],[2,3,4,5],[6,7,8],[9,10,11,12],[13,14]]
    line[1]=[[1,5],[0,4,8,12],[3,7,11],[2,6,10,14],[9,13]]
    line[2]=[[0,3,6,9],[1,4,7,10,13],[5,8,11,14]]
    return line
def distance(x_i):
    return np.linalg.norm(x_i[:-1]-x_i[1:],axis=1)




