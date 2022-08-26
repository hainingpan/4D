import argparse
import re
import h5py
import os
import numpy as np
import psutil
import time
def get_temp(fn):
    return int(re.search(r'.*(?=C)',fn).group(0))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('-i','--i',type=str,default='../STEM_recal/',help='working directory')
    parser.add_argument('-o','--o',type=str,default='data',help='output filename')
    args=parser.parse_args()
    workingdir=args.i
    output=args.o
    fn_list=[fn for fn in os.listdir(workingdir) if '.npy' in fn]
    temp_list=[get_temp(fn) for fn in fn_list]
    temp_list_sorted,fn_list_sorted=zip(*sorted(zip(temp_list,fn_list)))
    example=np.load(workingdir+fn_list_sorted[0])
    hf = h5py.File(workingdir+output+'.hdf5', 'w')
    hf.create_dataset('T',data=temp_list_sorted)
    arr = hf.create_dataset('I', (len(fn_list_sorted),*example.shape), chunks=True)
    # For compressed data
    # arr = hf.create_dataset('I', (len(fn_list_sorted),*example.shape), chunks=True,compression="gzip", compression_opts=9)
    for index,fn in enumerate(fn_list_sorted):
        print('Dumping {} (RAM={:.2f}GB)'.format(fn,psutil.Process().memory_info().rss / (1024 * 1024*1024)))
        st=time.time()
        arr[index]=np.load(workingdir+fn)
        print('Finish {} (RAM={:.2f}GB). Time {}'.format(fn,psutil.Process().memory_info().rss / (1024 * 1024*1024),time.time()-st))    
    
    hf.close()

