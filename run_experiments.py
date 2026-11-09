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



#DIRETORIO_PADRAO = 'D:\\Dropbox\\Projetos\\pessoal\\VQCA\\results\\reset\\'

DIRETORIO_PADRAO = 'C:\\Users\\petro\\Dropbox\\Projetos\\pessoal\\VQCA\\results\\'

#rule1 = [
#    [0, 0, 0, 1, 0, 0],
#    [0, 0, 1, 1, 1, 0],
#    [0, 1, 0, 1, 0, 1],
#    [1, 0, 0, 1, 0, 0],
#    [0, 0, 0, 1, 0, 0]
#]


#stats = VQCACircuitStatistics(n = 6, T = 1, operator = 21, path = DIRETORIO_PADRAO)

experiments = base.VQCAGlobalOptimizer(ca_patterns.rules, ga.VQCAOptimizerGA, path = DIRETORIO_PADRAO)
experiments.global_training()
experiments.fine_tunning(k = 30)

experiments = base.VQCAGlobalOptimizer(ca_patterns.rules, ga.VQCAOptimizerGA, path = DIRETORIO_PADRAO, noise = True)
experiments.global_training()
experiments.fine_tunning(k = 30)

experiments = base.VQCAGlobalOptimizer(ca_patterns.rules, ga.VQCAOptimizerGA, path = DIRETORIO_PADRAO, reset = True)
experiments.global_training()
experiments.fine_tunning(k = 30)

experiments = base.VQCAGlobalOptimizer(ca_patterns.rules, ga.VQCAOptimizerGA, path = DIRETORIO_PADRAO, reset = True, noise = True)
experiments.global_training()
experiments.fine_tunning(k = 30)


