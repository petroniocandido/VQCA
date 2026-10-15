from vqca import vqca
from vqca import ca_patterns
from vqca.optimizers import base, cobyla, cma, ga, spsa
from vqca.statistics import VQCACircuitStatistics

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


stats = VQCACircuitStatistics(n = 6, T = 1, operator = 21, path = DIRETORIO_PADRAO)

#optm = cobyla.VQCAOptimizerCOBYLA(pattern = np.array(ca_patterns.rule1), operator = 21)

#error = optm.mse(np.array(rule1))


print("SPSA padrão")

experiments = base.VQCAGlobalOptimizer(ca_patterns.rules, spsa.VQCAOptimizerSPSA, path = DIRETORIO_PADRAO, reset = False)
experiments.global_training()
experiments.fine_tunning()


print("SPSA noise")

experiments = base.VQCAGlobalOptimizer(ca_patterns.rules, spsa.VQCAOptimizerSPSA, path = DIRETORIO_PADRAO, noise = True)
experiments.global_training()
experiments.fine_tunning()

print("SPSA reset")

experiments = base.VQCAGlobalOptimizer(ca_patterns.rules, spsa.VQCAOptimizerSPSA, path = DIRETORIO_PADRAO, reset = True)
experiments.global_training()
experiments.fine_tunning()

print("SPSA reset noise")

experiments = base.VQCAGlobalOptimizer(ca_patterns.rules, spsa.VQCAOptimizerSPSA, path = DIRETORIO_PADRAO, reset = True, noise = True)
experiments.global_training()
experiments.fine_tunning()




