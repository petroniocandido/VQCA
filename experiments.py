from qnca import qnca
from qnca import ca_patterns
from qnca.optimizers import base, cobyla, cma, ga

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

dispositivo = 'GPU' if torch.cuda.is_available() else 'CPU'

DIRETORIO_PADRAO = 'D:\\Dropbox\\Projetos\\pessoal\\QNCA\\results\\'

#DIRETORIO_PADRAO = 'C:\\Users\\petro\\Dropbox\\Projetos\\pessoal\\QNCA\\results\\'

#rule1 = [
#    [0, 0, 0, 1, 0, 0],
#    [0, 0, 1, 1, 1, 0],
#    [0, 1, 0, 1, 0, 1],
#    [1, 0, 0, 1, 0, 0],
#    [0, 0, 0, 1, 0, 0]
#]


#optm = cobyla.QNCAOptimizerCOBYLA(pattern = np.array(ca_patterns.rule1), operator = 21)

#error = optm.mse(np.array(rule1))

#print(error)

experiments = base.QNCAGlobalOptimizer(ca_patterns.rules, cobyla.QNCAOptimizerCOBYLA, path = DIRETORIO_PADRAO, hadamard = True)
experiments.global_training()
experiments.fine_tunning()

#experiments = base.QNCAGlobalOptimizer(ca_patterns.rules, ga.QNCAOptimizerGA, path = DIRETORIO_PADRAO, shots = 100)
#experiments.global_training()
#experiments.fine_tunning(k = 30)


#experiments = base.QNCAGlobalOptimizer(ca_patterns.rules, cma.QNCAOptimizerCMA, path = DIRETORIO_PADRAO)
#experiments.global_training()
#experiments.fine_tunning()




#print(experiments.k_perturbed_best('28','30', k= 2))


