import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import h5py
import pickle
from matplotlib.animation import FuncAnimation
from GMM import GMM_model

def generate_ave_figure(I, fit_model, T_idx_max=None, filename=None, movie=False, figure=False, array=False):
    if T_idx_max is None:
        T_idx_max=I.shape[0]
    ij=fit_model.ij
    T_list=fit_model.rs['T']
    num_clusters=fit_model.k
    if -1 in ij.keys():
        cluster_list=list(range(num_clusters))+[-1]
    else:
        cluster_list=list(range(num_clusters))
    I_ave_cluster={}
    T_idx_list=np.arange(T_idx_max)
    for cluster_idx,cluster in enumerate(cluster_list):
        print(f'{cluster}')
        I_cluster=np.array([I[T_idx_list,i,j,:,:] for i,j in ij[cluster]]).mean(axis=0)
        I_ave_cluster[cluster]=I_cluster
    if figure:
        fig,axs=plt.subplots(T_idx_max,len(cluster_list),figsize=(15,3*T_idx_max),tight_layout=True)
    if array:
        data_store={}
    for T_idx in range(T_idx_max):
        for cluster_idx,cluster in enumerate(cluster_list):
            data=I_ave_cluster[cluster][T_idx]
            if array:
                if cluster!=-1:
                    data_store[(cluster,T_list[T_idx])]=data
            if figure:
                ax=axs[T_idx,cluster_idx]
                vmin,vmax=data.min(),data.max()
                im=ax.pcolormesh(data,cmap='gray', norm=colors.LogNorm(vmin=vmin, vmax=vmax))
                tit=ax.set_title('T={:d} K'.format(T_list[T_idx]))
                ax.set_xlabel(r'$k_x$')
                ax.set_ylabel(r'$k_y$')
        
    if filename is not None:
        if figure:
            fig.savefig(f'{filename}.png')
        if array:
            with open(f'{filename}.pickle','wb') as f:
                pickle.dump(data_store,f)

    if movie:
        fig_mp4,axs=plt.subplots(1,len(cluster_list),figsize=(15,3),tight_layout=True)
        T_idx=0
        im=[]
        tit=[]
        for cluster_idx,cluster in enumerate(cluster_list):
            ax=axs[cluster_idx]
            data=I_ave_cluster[cluster][T_idx]
            vmin,vmax=data.min(),data.max()
            im.append(ax.pcolormesh(data,cmap='gray', norm=colors.LogNorm(vmin=vmin, vmax=vmax)))
            tit.append(ax.set_title('T={:d} K'.format(T_list[T_idx])))
            cluster_text='Vacuum' if cluster==-1 else f'Cluster {cluster}'
            ax.text(0,1,cluster_text,ha='right',va='bottom',transform=ax.transAxes)
            
            ax.set_xlabel(r'$k_x$')
            ax.set_ylabel(r'$k_y$')
        
        def animate(i):
            for cluster_idx,cluster in enumerate(cluster_list):
                data=I_ave_cluster[cluster][i]
                im[cluster_idx].set_array(data)
                tit[cluster_idx].set_text('T={:d} K'.format(T_list[i]))
            return im+tit

        anim = FuncAnimation(fig_mp4, animate, interval=1000, frames=T_idx_max,blit=True,repeat=True)
        anim.save(f'{filename}.mp4',dpi=300)

def main():
    workingdir = '/share/kim/STEM_sample_2/'
    f = h5py.File(workingdir + 'data.hdf5', 'r')
    I, T = f['I'], f['T']

    with open('count_pts_outlier_auto.pickle', 'rb') as f:
        count_pts_outlier_auto = pickle.load(f)

    generate_ave_figure(I, fit_model=count_pts_outlier_auto, T_idx_max=None, filename='ave_cluster_real_space_auto_no_vac', array=True)

if __name__ == '__main__':
    main()
