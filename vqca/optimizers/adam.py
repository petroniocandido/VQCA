from qiskit_algorithms.optimizers import ADAM
import numpy as np
from .base import VQCAOptimizer

class VQCAOptimizerADAM(VQCAOptimizer):
  name = 'ADAM'
  def __init__(self, **kwargs):
    super(VQCAOptimizerADAM, self).__init__(**kwargs)

  def training_loop(self, param = None):

    optimizer = ADAM(maxiter=1000, lr=0.01)
    
    parametros = np.random.rand(self.num_param) * (2 * np.pi) if param is None else param

    objetivo = lambda x: self.funcao_custo(x)

    result = result = optimizer.minimize(fun=objetivo, x0=parametros)

    return result