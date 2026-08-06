from Individual import Individual, Worker, Job, Gene, WorkerJobPreference
import random
import copy
import matplotlib.pyplot as plt
import time
import numpy as np
import cProfile
import pstats
from GeneticAlgorithmOptimized import GA

from randomize_data import randomize_ja_data

from DataFile import worker_datas, job_datas

def test_ga(worker_size, job_size, pref_size, CRSO, MUT, ELT, POP, GEN, dbg):
    execution_times = []
    
    worker_size_list = [size for size in range(1, worker_size+1)]
    for size in worker_size_list:
        print(f"input size : {size}")
        wdatas, jdatas = randomize_ja_data(size, job_size, pref_size)
        start = time.time()
        GA(wdatas, jdatas, CRSO, MUT, ELT, POP, GEN, dbg)
        end = time.time()
        execution_time = end-start
        print("Execution Time :", f"{execution_time:.6f}s\n")
        execution_times.append(execution_time)

    plt.plot(worker_size_list, execution_times, marker='o')

    m, b = np.polyfit(worker_size_list, execution_times, 1)
    plt.plot(worker_size_list, m*np.array(worker_size_list)+b)
    
    plt.title("Execution Time vs. Input Size")
    plt.xlabel(f"Worker size(N)\nTest Data Parameter: Worker Size({worker_size}), Job Size({job_size}), Preferences Size({pref_size})\nTest Parameter: Population Size({POP}), Generation({GEN}), CRSO({CRSO}), MUT({MUT}), ELT({ELT})")
    plt.ylabel("Execution Time (s)")
    plt.grid(True)
    plt.show()
    

if __name__ == "__main__":
    worker_size = 100
    job_size = 50 # must be >= pref_size
    pref_size = 30

    # GA PARAMETER

    CRSO = 0.45
    MUT = 0.4 
    ELT = 0.2
    POP = 10
    GEN = 500
    test_ga(worker_size, job_size, pref_size, CRSO, MUT, ELT, POP, GEN, False)

    


