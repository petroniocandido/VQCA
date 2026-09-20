from vqca import vqca
from vqca import ca_patterns
from vqca.optimizers import base, cobyla, cma, ga

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

dispositivo = 'GPU' if torch.cuda.is_available() else 'CPU'

DIRETORIO_PADRAO = 'D:\\Dropbox\\Projetos\\pessoal\\QNCA\\results\\'

#DIRETORIO_PADRAO = 'C:\\Users\\petro\\Dropbox\\Projetos\\pessoal\\QNCA\\results\\'

experiments = base.VQCAGlobalOptimizer(ca_patterns.rules, cobyla.VQCAOptimizerCOBYLA, path = DIRETORIO_PADRAO)
experiments.plot_results()