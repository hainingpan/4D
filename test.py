import time
from mpi4py.futures import MPIPoolExecutor

def f(x):
    i=0
    s=0
    for i in range(200000):
        s+=i
    return s

if __name__=='__main__':
    st=time.time()
    executor=MPIPoolExecutor()
    pool=executor.map(f,range(100000))
    s_list=[]
    for r_i,result in enumerate(pool):
        s_list.append(result)
    executor.shutdown()

    # for i in range(0,101,25):
    #     print("\r>>TESTING - {:0>3d}%".format(i), end='', flush=True)
    #     time.sleep(1)
    # print()

    print(time.time()-st)




