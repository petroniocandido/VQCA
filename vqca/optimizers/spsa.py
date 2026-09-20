import numpy as np
#import spsa
from qiskit_algorithms.optimizers import SPSA
from .base import VQCAOptimizer

class VQCAOptimizerSPSA(VQCAOptimizer):
  name = 'SPSA'
  def __init__(self, **kwargs):
    super(VQCAOptimizerSPSA, self).__init__(**kwargs)
    self.spsa = SPSA(maxiter=300)

  def training_loop(self, param = None):
    
    parametros = np.random.rand(self.num_param) * (2 * np.pi) if param is None else param

    objetivo = lambda x: self.funcao_custo(x)

    # Run classical optimization (e.g., COBYLA or SLSQP)
    #result = spsa.minimize(objetivo, parametros.tolist())

    result = self.spsa.minimize(objetivo, parametros)

    print(result)

    return result