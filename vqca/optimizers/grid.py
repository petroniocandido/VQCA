import numpy as np
from itertools import product
from .base import VQCAOptimizer

class VQCAOptimizerGridSearch(VQCAOptimizer):
  name = 'GridSearch'
  def __init__(self, **kwargs):
    super(VQCAOptimizerGridSearch, self).__init__(**kwargs)
    self.bins = kwargs.get('bins',10)

  def training_loop(self, param = None):
    
    iteradores = [[k for k in np.linspace(0,2*np.pi, self.bins)] for j in range(self.num_param)]

    for k in product(*iteradores):
       erro = self.funcao_custo(np.array(k))
       print("MSE: {} PARAM: {}".format(erro, k))       
    
    return None
