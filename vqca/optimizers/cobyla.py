import numpy as np
from scipy.optimize import minimize
from .base import VQCAOptimizer

class VQCAOptimizerCOBYLA(VQCAOptimizer):
  name = 'COBYLA'
  def __init__(self, **kwargs):
    super(VQCAOptimizerCOBYLA, self).__init__(**kwargs)

  def training_loop(self, param = None):
    
    parametros = np.random.rand(self.num_param) * (2 * np.pi) if param is None else param

    objetivo = lambda x: self.funcao_custo(x)

    # Run classical optimization (e.g., COBYLA or SLSQP)
    result = minimize(objetivo, parametros, method='COBYLA')

    return result
