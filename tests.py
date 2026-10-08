from vqca import vqca
from vqca import ca_patterns
from vqca.optimizers import base, cobyla, cma, ga, spsa

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

dispositivo = 'GPU' if torch.cuda.is_available() else 'CPU'

print(dispositivo)



DIRETORIO_PADRAO = 'D:\\Dropbox\\Projetos\\pessoal\\VQCA\\results\\reset\\'

#DIRETORIO_PADRAO = 'C:\\Users\\petro\\Dropbox\\Projetos\\pessoal\\VQCA\\results\\reset\\'

#rule1 = [
#    [0, 0, 0, 1, 0, 0],
#    [0, 0, 1, 1, 1, 0],
#    [0, 1, 0, 1, 0, 1],
#    [1, 0, 0, 1, 0, 0],
#    [0, 0, 0, 1, 0, 0]
#]


#optm = cobyla.VQCAOptimizerCOBYLA(pattern = np.array(ca_patterns.rule1), operator = 21)

#error = optm.mse(np.array(rule1))


print("COBYLA")

experiments = base.VQCAGlobalOptimizer(ca_patterns.rules, cobyla.VQCAOptimizerCOBYLA, path = DIRETORIO_PADRAO, iqp = True)
experiments.global_training()
experiments.fine_tunning()


print("GA")

experiments = base.VQCAGlobalOptimizer(ca_patterns.rules, ga.VQCAOptimizerGA, path = DIRETORIO_PADRAO, shots = 100, iqp = True)
experiments.global_training()
experiments.fine_tunning(k = 30)



