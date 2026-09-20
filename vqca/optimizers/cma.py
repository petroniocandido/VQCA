import cma
import numpy as np
from .base import VQCAOptimizer

class VQCAOptimizerCMA(VQCAOptimizer):
  name = 'CMA'
  def __init__(self, **kwargs):
    super(VQCAOptimizerCMA, self).__init__(**kwargs)
    self.maxiter = kwargs.get('maxiter', 60)
    self.popsize = kwargs.get('population', 15)

  def training_loop(self, param = None):
    parametros = [np.pi/2] * self.num_param  if param is None else param

    objetivo = lambda x: self.funcao_custo(x)

    options = {'maxiter': self.maxiter, 'popsize': self.popsize, 'bounds': [0, 2 * np.pi]}

    sigma0 = .5        # Desvio padrão inicial (tamanho do passo de busca)

    es = cma.fmin2(objetivo, x0=parametros, sigma0 = sigma0, options = options)

    return es
