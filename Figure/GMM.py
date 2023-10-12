import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import FunctionTransformer,StandardScaler
from sklearn.pipeline import Pipeline
import pickle
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV
from matplotlib.animation import FuncAnimation
class GMM_model:
    def __init__(self,filename,mask=None,mask_T=None):
        '''
        rs: (T_idx,x_idx,y_idx)
        mask: ((x_min,x_max),(y_min,y_max))
        '''
        self.filename=filename
        self.mask=mask if mask is not None else slice(None)
        self.mask_T=mask_T if mask_T is not None else slice(None)
        self.rs=self.load()
        self.labels_={}
        self.X_orig={}
        self.X={}
        self.bic={}
        self.silhouette={}
        self.search={}
        self.labels_grid={}

    def load(self):
        with open(self.filename,'rb') as f:
            rs=pickle.load(f)
        # if self.mask is not None:
        #     for key in ['mean','std','length']:
        #         rs[key]=rs[key][:,self.mask[0][0]:self.mask[0][1],self.mask[1][0]:self.mask[1][1]]
        #     rs['x']=np.arange(self.mask[0][0],self.mask[0][1])
        #     rs['y']=np.arange(self.mask[1][0],self.mask[1][1])
        if self.mask is None:
            self.mask=(np.ones(rs['mean'].shape[1:])==1)
        rs['T']=rs['T']+273
        return rs

    def preprocessing(self,kind):
        self.X_orig[kind]=self.rs[kind][:,self.mask][self.mask_T].T

        # self.X_orig[kind]=self.rs[kind].reshape((self.rs[kind].shape[0],-1)).T

        # estimators=[('standardize',StandardScaler()),]
        estimators=[('T',FunctionTransformer(lambda x:x.T)),('standardize',StandardScaler()),('T2',FunctionTransformer(lambda x:x.T)),]

        pipeline=Pipeline(estimators)
        self.X[kind]=pipeline.fit_transform(self.X_orig[kind])
        
    def GMM_fit(self,kind,k=3):
        self.preprocessing(kind)
        self.k=k
        self.gm=GaussianMixture(n_components=self.k)
        self.labels_[kind]=self.gm.fit_predict(self.X[kind])
        
    
    def GMM_hypertuning(self,kind,k_range=np.arange(2,7),trial=10):
        self.bic[kind]=pd.DataFrame()
        self.silhouette[kind]=pd.DataFrame()
        if not kind in self.X:
            self.preprocessing(kind)
        for k in k_range:
            bic_k=[]
            silhouette_k=[]
            for i in range(trial):
                print('\r k={} i={}'.format(k,i),end='',flush=True)
                self.GMM_fit(kind,k)
                self.gm.fit(self.X[kind])    
                bic_k.append(self.gm.bic(self.X[kind]))
                silhouette_k.append(silhouette_score(self.X[kind],self.gm.fit_predict(self.X[kind])))
            self.bic[kind][k]=bic_k
            self.silhouette[kind][k]=silhouette_k
    
    def GMM_hypertuning_GridSearchCV(self,kind,k_range=np.arange(2,7),trial=10):
        if not kind in self.search:
            if not kind in self.X:
                self.preprocessing(kind)
            # def scoring(clf,X):
            #     labels_=clf.fit_predict(self.X[kind])
            #     return {'bic':clf.bic(self.X[kind]),'silhouette':silhouette_score(self.X[kind], labels_)}
            
            self.k_range=k_range
            self.search[kind]=GridSearchCV(GaussianMixture(n_init=trial), param_grid={"n_components":k_range},n_jobs=-1,return_train_score=True,scoring=scoring,verbose=4,cv=[(slice(None), slice(None))],refit=False)
            self.search[kind].fit(self.X[kind])


    def GMM_plot(self,kind,permutation=None,ax=None,**kwargs):
        if permutation is None:
            permutation=np.arange(self.k)
        offset=permutation-np.arange(self.k)
        if ax is None:
            fig, ax= plt.subplots(1,3,figsize=(13,4),tight_layout=True)
        cmap = plt.get_cmap('plasma', self.k)
        bounds = np.arange(self.k+1)-0.5
        norm = colors.BoundaryNorm(bounds, cmap.N)
        labels_=self.labels_[kind]
        labels_offset=np.array([offset[labels_[idx]] for idx in range(labels_.shape[0])])
        labels_=labels_+labels_offset
        cnt=0
        self.labels_grid[kind]=np.zeros(self.rs['mean'].shape[1:],)
        for i in self.rs['x']:
            for j in self.rs['y']:
                if self.mask[i,j]:
                    self.labels_grid[kind][i,j]=labels_[cnt]
                    cnt+=1
                else:
                    self.labels_grid[kind][i,j]=np.nan

        
        im=ax[0].imshow(self.labels_grid[kind],cmap=cmap,norm=norm,extent=(self.rs['y'][0],self.rs['y'][-1],self.rs['x'][-1],self.rs['x'][0]))
        cb=plt.colorbar(im,ax=ax[0],ticks=np.arange(self.k),boundaries=bounds)
        cb.ax.set_yticklabels([f'{i}' for i in range(self.k)])
        bic_text='BIC={:.0f}, Silhouette={:.2f}'.format(self.gm.bic(self.X[kind]),silhouette_score(self.X[kind],labels_))
        for idx in range(self.k):
            bic_text=bic_text+'\n{}:{}'.format(idx,np.sum(labels_==idx))
        ax[0].set_title(bic_text)
        ax[0].set_xlabel('y')
        ax[0].set_ylabel('x')

        for ax_idx in range(1,3):
            color_list=[cmap(idx) for idx in range(self.k)]
            for n,color in zip(range(self.k),color_list):
                mean=self.X[kind][labels_==n].T.mean(axis=-1) if ax_idx==1 else self.X_orig[kind][labels_==n].T.mean(axis=-1)
                ax[ax_idx].plot(self.rs['T'][self.mask_T],mean,color=color,label=str(n),**kwargs)
                error=self.X[kind][labels_==n].T.std(axis=-1) if ax_idx==1 else self.X_orig[kind][labels_==n].T.std(axis=-1)
                ax[ax_idx].fill_between(self.rs['T'][self.mask_T],mean-error,mean+error,color=color,alpha=0.5)
            ax[ax_idx].set_xlabel('T (K)')
            if ax_idx==1:
                ax[ax_idx].set_ylabel('z-score of {}'.format(kind))
            else:
                ax[ax_idx].set_ylabel(kind)
            ax[ax_idx].legend()

    def GMM_plot_single(self,kind,cluster_list,cond=None,errorbar=True,ax=None,ylim=None,**kwargs):
        if ax is None:
            fig,ax=plt.subplots()
        labels_=self.labels_[kind]

        cmap = plt.get_cmap('plasma', self.k)
        color_list=[cmap(idx) for idx in range(self.k)]
        for n in cluster_list:
            mean=self.X_orig[kind][labels_==n].T.mean(axis=-1)
            ax.plot(self.rs['T'][self.mask_T],mean,color=color_list[n],label=str(n),**kwargs)
            if errorbar:
                error=self.X_orig[kind][labels_==n].T.std(axis=-1)
                ax.fill_between(self.rs['T'][self.mask_T],mean-error,mean+error,color=color_list[n],alpha=0.5)
        ax.set_xlabel('T (K)')
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_ylabel(kind)
        ax.legend()
        if cond is not None:
            ax2=ax.twinx()
            ax2.plot(cond[:,0],cond[:,1],color='k',ls='dashed')
            ax2.set_xlim(self.rs['T'][self.mask_T][0],self.rs['T'][self.mask_T][-1])
            ax2.set_ylabel('R ($\Omega$)')
            ax2.tick_params(axis='y', labelcolor='k')

    def convert_real_space(self,kind):
        labels_grid=self.labels_grid[kind]
        self.ij={}
        for i in (self.rs['x']):
            for j in (self.rs['y']):
                label=-1 if np.isnan(labels_grid[i,j]) else labels_grid[i,j]
                if label in self.ij:
                    self.ij[label].append((i,j))
                else:
                    self.ij[label]=[]
        
        # for cluster in range(self.k):
        #     idx,=np.where(labels_==cluster)
        #     self.ij[cluster]=np.unravel_index(idx, self.rs[kind].shape[1:])
        return self.ij

def scoring(clf,X):
    labels_=clf.fit_predict(X)
    return {'bic':clf.bic(X),'silhouette':silhouette_score(X, labels_)}
