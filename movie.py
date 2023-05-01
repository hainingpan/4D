import argparse
import h5py
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import colors
from GMM import GMM_model

def generate_ave_figure(I, fit_model, T_idx_max=None, filename=None, movie=False, figure=False, array=False, filetype_list=['mp4'], dpi=100, mask_k=None):
    # Your function implementation here
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
        I_cluster[:,~mask_k]=np.nan
        I_ave_cluster[cluster]=I_cluster
        # I_ave_cluster[cluster]=I_cluster[:,mask_k]
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
                vmin,vmax=np.nanmin(data),np.nanmax(data)
                im=ax.pcolormesh(data,cmap='gray', norm=colors.LogNorm(vmin=vmin, vmax=vmax))
                tit=ax.set_title('T={:d} K'.format(T_list[T_idx]))
                ax.set_xlabel(r'$k_x$')
                ax.set_ylabel(r'$k_y$')
                # ax.set_aspect('equal')
        
    if filename is not None:
        if figure:
            fig.savefig(f'{filename}.png',dpi=1000)
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
            vmin,vmax=np.nanmin(data),np.nanmax(data)
            im.append(ax.pcolormesh(data,cmap='gray', norm=colors.LogNorm(vmin=vmin, vmax=vmax)))
            tit.append(ax.set_title('T={:d} K'.format(T_list[T_idx])))
            cluster_text='Vacuum' if cluster==-1 else f'Cluster {cluster}'
            ax.text(0,1,cluster_text,ha='right',va='bottom',transform=ax.transAxes)
            
            ax.set_xlabel(r'$k_x$')
            ax.set_ylabel(r'$k_y$')
            # ax.set_aspect('equal')
        
        def animate(i):
            for cluster_idx,cluster in enumerate(cluster_list):
                data=I_ave_cluster[cluster][i]
                im[cluster_idx].set_array(data)
                tit[cluster_idx].set_text('T={:d} K'.format(T_list[i]))
            return im+tit

        anim = FuncAnimation(fig_mp4, animate, interval=1000, frames=T_idx_max,blit=True,repeat=True)
        for ext in filetype_list:
            anim.save(f'{filename}.{ext}',dpi=dpi)

def main(args):
    with open(args.model_filename, 'rb') as f:
        fit_model = pickle.load(f)

    workingdir='/share/kim/STEM_sample_3/'
    f = h5py.File(workingdir+'data.hdf5', 'r')
    I, T = f['I'], f['T']

    mask_k = np.full(I.shape[-2:], True)
    mask_k[:, 85:] = False

    generate_ave_figure(I, fit_model=fit_model, T_idx_max=args.T_idx_max, filename=args.filename, array=args.array, movie=args.movie, figure=args.figure, mask_k=mask_k, filetype_list=args.filetype_list, dpi=args.dpi)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--filename', type=str, help='Filename for output')
    parser.add_argument("--T_idx_max", default=None, type=int,help='T index maximum value')
    parser.add_argument('--model_filename', type=str, help='Filename for GMM_model')
    parser.add_argument('--movie', action='store_true', help='Generate movie')
    parser.add_argument('--figure', action='store_true', help='Generate figure')
    parser.add_argument('--array', action='store_true', help='Generate array')
    parser.add_argument('--filetype_list', nargs='+', default=['mp4'], help='List of file types for output')
    parser.add_argument('--dpi', type=int, default=100, help='DPI for output files')

    args = parser.parse_args()
    main(args)
