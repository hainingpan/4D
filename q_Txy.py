import numpy as np
from skimage.feature import peak_local_max
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from mpi4py.futures import MPIPoolExecutor
from mpi4py import MPI 
import time
import pickle
import argparse
import h5py


from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from q_stat import *
        
def parallel(params):
    """
    Executes the specified q_stat method in parallel.

    Parameters
    ----------
    params : tuple
        A tuple containing the necessary parameters for the q_stat method.

    Returns
    -------
    result : object
        The result of the q_stat method execution.
    """
    T_idx,x,y,workingdir,auto,kind,outlier=params
    q=q_stat(T_idx,x,y,workingdir,auto)
    if 'all_peaks' == kind:
        return q.all_peaks(remove_outlier=outlier,threshold=3)
    if 'remove_bragg' == kind:
        return q.remove_bragg(remove_outlier=outlier)
    if 'count_pts' == kind:
        return q.count_pts(repel=np.sqrt(2),remove_central_shell=outlier)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-i','--i',type=str,default='/share/kim/STEM_sample_2/',help='working directory')
    parser.add_argument('-o','--o',type=str,help='output filename')
    parser.add_argument('-drop','--drop',type=int,default=1,help='drop the last # temperatures')
    parser.add_argument('-kind','--kind',type=str,help='kind=all_peaks|remove_bragg')
    parser.add_argument('-outlier','--outlier',action='store_true',help='whether to remove outlier. default: False')
    parser.add_argument('-auto','--auto',action='store_true',help='whether to generate Bragg peaks each time. default: False')
    
    args=parser.parse_args()
    workingdir=args.i
    output=args.o
    kind=args.kind
    outlier=args.outlier

    st=time.time()
    f=h5py.File(workingdir+'data.hdf5','r')
    I,T=f['I'],f['T']

    comm = MPI.COMM_WORLD
    print('Nodes: {:d}'.format(comm.Get_size()))
    
    T_list=np.arange(I.shape[0]-args.drop)
    x_list=np.arange(I.shape[1])
    y_list=np.arange(I.shape[2])

    para_list=[(T_idx,x,y,workingdir,args.auto, kind,outlier) for T_idx in T_list for x in x_list for y in y_list]
    executor=MPIPoolExecutor()
    pool=executor.map(parallel,para_list)
    # pool=map(parallel,para_list)

    q_mean_map=np.zeros((T_list.shape[0],x_list.shape[0],y_list.shape[0]))
    q_std_map=np.zeros((T_list.shape[0],x_list.shape[0],y_list.shape[0]))
    q_length_map=np.zeros((T_list.shape[0],x_list.shape[0],y_list.shape[0]))
    bragg_peaks=np.zeros((T_list.shape[0],x_list.shape[0],y_list.shape[0],17,2))
    count=0
    for r_i,result in zip(para_list,pool):
        T_idx,x,y,_,_,_,_=r_i
        x-=x_list[0]
        y-=y_list[0]
        q_mean_map[T_idx,x,y]=result['mean']
        q_std_map[T_idx,x,y]=result['std']
        q_length_map[T_idx,x,y]=result['length']
        bragg_peaks[T_idx,x,y]=result['bragg']
        count+=1
        if count%100==0:
            duration=time.time()-st
            print('\rdone:{}/{} Time elapsed:{:.1f}s Expected remaining: {:.1f}s Expected total: {:.1f}s'.format(count,T_list.shape[0]*x_list.shape[0]*y_list.shape[0],duration,duration*((T_list.shape[0]*x_list.shape[0]*y_list.shape[0])/count-1),duration/(count/(T_list.shape[0]*x_list.shape[0]*y_list.shape[0]))),end='',flush=True)      
    print()
    executor.shutdown()

    with open('q_Txy_{}{}_{}.pickle'.format(kind,'_outlier'*outlier,output),'wb') as f:
        pickle.dump({'mean':q_mean_map,'std':q_std_map,'length':q_length_map,'x':x_list,'y':y_list,'T_list':T_list,'T':T[T_list],'bragg_peaks':bragg_peaks,'args':args},f)
    print(time.time()-st)