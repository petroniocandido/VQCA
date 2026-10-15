from vqca.statistics import VQCACircuitStatistics

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

dispositivo = 'GPU' if torch.cuda.is_available() else 'CPU'

print(dispositivo)



#DIRETORIO_PADRAO = 'D:\\Dropbox\\Projetos\\pessoal\\VQCA\\results\\'

DIRETORIO_PADRAO = 'C:\\Users\\petro\\Dropbox\\Projetos\\pessoal\\VQCA\\results\\'


stats = VQCACircuitStatistics(n = 6, T = 1, operator = 21, hadamard = True, path = DIRETORIO_PADRAO, seed=47)

stats.run()
