import numpy as np
from scipy import ndimage
from sklearn.neighbors import NearestNeighbors
from collections import deque
import matplotlib.pyplot as plt
from matplotlib import colors
class COM:
    def __init__(self,data,k0=[54,54],k1=[51,83],k2=[28,65],
    i_range=np.array([[-2,2],[-1,2],[0,2],
                    [-2,1],[-1,1],[0,1],[1,1],
                    [-1,0],[0,0],[1,0],
                    [-1,-1],[0,-1],[1,-1],[2,-1],
                    [0,-2],[1,-2],[2,-2]]),crop=None,radius=3):
        self.data=data
        self.k0=np.array(k0)
        self.k1=np.array(k1)
        self.k2=np.array(k2)
        self.i_range=i_range
        self.crop=np.full(data.shape, True) if crop is None else crop
        self.radius=radius
        self.x_i=self.generate_x_i()
        self.lines=generate_lines(index_range=self.i_range,vectors=np.array([[1,0],[0,1],[-1,1]]))

    def generate_x_i(self):
        x_i=self.i_range@np.array([self.k1-self.k0,self.k2-self.k0])+self.k0
        x_i=np.round(x_i).astype(int)
        uncropped_pts=(self.crop[tuple(x_i.T)])
        self.i_range=self.i_range[uncropped_pts]
        x_i=x_i[uncropped_pts]
        # x_i=self.relocate_max(x_i)
        return x_i
    
    # def relocate_max(self,x_i_list):
    #     x_i_updated=np.zeros_like(x_i_list)
    #     for label_idx,x_i in enumerate(x_i_list):
    #         i_min,j_min=max(x_i[0]-self.radius,0),max(x_i[1]-self.radius,0)
    #         i_max,j_max=min(x_i[0]+self.radius+1,self.data.shape[0]),min(x_i[1]+self.radius+1,self.data.shape[1])
    #         data=self.data[i_min:i_max,j_min:j_max]
    #         idx=np.unravel_index(data.argmax(), data.shape)
    #         x_i_updated[label_idx]=np.arange(i_min,i_max)[idx[0]],np.arange(j_min,j_max)[idx[1]]
    #     return x_i_updated

    # def generate_mask(self):
    #     # if x_i is None:
    #     #     x_i=self.generate_x_i()
    #     mask=np.zeros(self.data.shape,dtype=int)
    #     x_i_all=[]
    #     for i in range(-self.radius,self.radius+1):
    #         for j in range(-self.radius,self.radius+1):
    #             x_i_all.append(self.x_i+np.array([i,j]))
    #     x_i_all=np.vstack(x_i_all)
    #     x_i_all[x_i_all<0]=0
    #     x_i_all[x_i_all[:,0]>=self.data.shape[0]]=self.data.shape[0]-1
    #     x_i_all[x_i_all[:,1]>=self.data.shape[1]]=self.data.shape[1]-1
    #     mask[tuple(x_i_all.T)]=1
    #     return mask
    def generate_mask_ordered(self,x_i_list):
        mask=np.zeros(self.data.shape,dtype=int)
        data=np.zeros(self.data.shape,dtype=float)
        x_i_list=np.round(x_i_list).astype(int)
        for label_idx,x_i in enumerate(x_i_list):
            i_min,j_min=max(x_i[0]-self.radius,0),max(x_i[1]-self.radius,0)
            i_max,j_max=min(x_i[0]+self.radius+1,self.data.shape[0]),min(x_i[1]+self.radius+1,self.data.shape[1])
            mask[i_min:i_max,j_min:j_max]=label_idx+1
            data[i_min:i_max,j_min:j_max]=self.data[i_min:i_max,j_min:j_max]-self.data[i_min:i_max,j_min:j_max].min()

        return data,(mask,label_idx+1)

    def generate_com(self):
        x_i_old=self.x_i
        data,lbl=self.generate_mask_ordered(x_i_old)
        x_i_new=np.array(ndimage.center_of_mass(data,lbl[0],range(1,lbl[1]+1)))
        update=np.linalg.norm(x_i_new-x_i_old,axis=1).max()
        while update>0.1:
            x_i_old=x_i_new
            data,lbl=self.generate_mask_ordered(x_i_old)
            x_i_new=np.array(ndimage.center_of_mass(data,lbl[0],range(1,lbl[1]+1)))
            update=np.linalg.norm(x_i_new-x_i_old,axis=1).max()
        return x_i_new

    def visualize(self):
        com_ij=self.generate_com()
        fig,ax=plt.subplots(figsize=(6,6))
        vmin,vmax=np.nanmin(self.data),np.nanmax(self.data)
        im=ax.imshow(self.data,cmap='gray', norm=colors.LogNorm(vmin=vmin, vmax=vmax))
        ax.set_xlabel(r'$k_x$')
        ax.set_ylabel(r'$k_y$')
        ax.set_aspect('equal')

        ax.scatter(com_ij[:,1],com_ij[:,0],color='b',s=3)
        [ax.text(com_ij[idx,1],com_ij[idx,0],f'{idx}',color='b') for idx in range(com_ij.shape[0])]


    def average_dist(self):
        com_i=self.generate_com()
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(com_i)
        distances, indices = nbrs.kneighbors(com_i)
        return distances[:,1].mean()

def average_dist_direction(com_i,lines):
    dist={}
    ave_dist=[]
    for key,val in lines.items():
        dist[key]=[]
        for pts_idx in val:
            pts=com_i[pts_idx]
            dist[key].append(distance(pts))
    # return dist
    for key,val in dist.items():
        ave_distance=np.hstack(dist[key]).mean()
        ave_dist.append(ave_distance)
    return np.array(ave_dist)



def generate_lines(index_range, vectors):
    """
    Generate a list of lines for given vectors.

    Parameters
    ----------
    index_range : numpy.ndarray
        2D array containing the indices.
    vectors : numpy.ndarray
        Array of vectors along an orientation are to be generated.

    Returns
    -------
    dict
        A dictionary where each key corresponds to a vector orientation and each value is a list of lines.

    """
    lines={}
    for vector_idx in range(3):
        visited = [False] * index_range.shape[0]
        lines[vector_idx] = [extend(point_idx, vectors[vector_idx], index_range, visited) for point_idx in range(index_range.shape[0]) if not visited[point_idx]]
    return lines


def extend(start_point_idx, vector, index_range, visited):
    """
    Extend the line from the starting point in the direction of the vector.

    Parameters
    ----------
    start_point_idx : int
        Starting point index.
    vector : numpy.ndarray
        Vector to extend the line.
    index_range : numpy.ndarray
        2D array containing the indices.
    visited : list
        A list of boolean values indicating which indices have been visited.

    Returns
    -------
    collections.deque
        A deque containing indices of the extended line.

    """
    point_idx_list = deque([start_point_idx])
    visited[start_point_idx] = True
    for direction in [-1,1]:
        next_point = index_range[start_point_idx] + direction* vector
        idx = is_contained(next_point, index_range, visited)
        while idx is not None: 
            if direction == 1:
                point_idx_list.append(idx)
            else:
                point_idx_list.appendleft(idx)
            visited[idx] = True
            next_point = next_point + direction*vector
            idx = is_contained(next_point, index_range, visited)

    # next_point = index_range[start_point_idx] - vector
    # idx = is_contained(next_point, index_range, visited)
    # while idx != -1:
    #     point_idx_list.appendleft(idx)
    #     visited[idx] = True
    #     next_point = next_point - vector
    #     idx = is_contained(next_point, index_range, visited)

    return point_idx_list

def is_contained(point, index_range, visited):
    """
    Check whether a point is contained in the index range and not visited.

    Parameters
    ----------
    point : numpy.ndarray
        Point to be checked.
    index_range : numpy.ndarray
        2D array containing the indices.
    visited : list
        A list of boolean values indicating which indices have been visited.

    Returns
    -------
    int
        The index of the point if it is contained in the index range and not visited, else None.

    """
    idx = np.where((point == index_range).all(axis=1))[0]
    return idx[0] if idx.size > 0 and not visited[idx[0]] else None


def distance(x_i):
    return np.linalg.norm(x_i[:-1]-x_i[1:],axis=1)
