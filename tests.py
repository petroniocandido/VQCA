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

experiments = base.QNCAGlobalOptimizer(ca_patterns.rules, cobyla.QNCAOptimizerCOBYLA, path = DIRETORIO_PADRAO)
experiments.plot_results()