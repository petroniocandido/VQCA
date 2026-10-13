from qiskit import QuantumCircuit
from qiskit_aer import  Aer, AerSimulator
from qiskit_aer.noise import  NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from qiskit import transpile
import numpy as np
import torch
from vqca.vqca import VQCA
from vqca.operators import VQCAOperator, get_id


class VQCAEstimator(object):
  """
    VQCA Estimator class.

    Generate the (n x T) output pattern of a VQCA circuit filled with the informed variational parameters and the other structural VQCA parameters,
    given the informed initial state.
    
  
    Attributes:
      n (int): Number of cellular automata cells
      T (int): Number of simulation iterations
      initial (string): Initial state of the cellular automada
      operator (VQCAOperator): Cell state transition circuit
      num_param (int)
      hadamard (bool): If the hadamard transform should be applied in each cell state transtition circuit
      iqp (bool): If an iqp preparing should be applied in each cell state transtition circuit
      reset (bool): If the reset gate should be applied at the end of each iteration.  Attention! This parameter makes the VQCA an irreversible circuit
      noise (bool)
      shots (int)
      backend (qiskit.Backend): The Qiskit execution backend.      
      noise_model
      
    """
  def __init__(self, **kwargs):
    self.hadamard = kwargs.get('hadamard',False)    
    self.iqp = kwargs.get('iqp',False)
    self.reset = kwargs.get('reset',False)
    self.kwargs = kwargs
    self.T = kwargs.get('T',False)
    self.n = kwargs.get('n',False)
    self.initial = kwargs.get('initial',[])
    self.device = 'GPU' if torch.cuda.is_available() else 'CPU'
    self.operator = kwargs['operator']
    
    if isinstance(self.operator, int):
      self.operator = get_id(self.n, self.operator)
      self.num_param = self.operator.npar 
    elif isinstance(self.operator, VQCAOperator):
      self.num_param = self.operator.npar 
    else:
      raise Exception("Operator not informed or type not recognized.")
    
    self.shots = kwargs.get('shots',33)

    self.noise = kwargs.get('noise',False)

    if self.noise:
      backend = FakeSherbrooke()
      self.noise_model = NoiseModel.from_backend(backend)

    if 'backend' in kwargs:
      self.backend = kwargs['backend']
    else:      
        self.backend = Aer.get_backend('qasm_simulator', device=self.device)

  def output(self, parameters):
    """
    Generate the (n x T) output pattern of a VQCA circuit filled with the informed variational parameters and the other structural VQCA parameters,
    given the informed initial state.

    Args:
      parameters (numpy.ndarray) :  variational parameters

    Returns:
      numpy.ndarray : An n*T matrix with the generated pattern, given the initial

    """
    evolution = np.zeros((self.T,self.n))

    for t in range(self.T-1):
      qc = VQCA(n = self.n, operator = self.operator, initial=self.initial, T = t+1, iqp = self.iqp, \
                backend = self.backend, parameters = parameters, hadamard = self.hadamard, reset = self.reset)

      if self.noise:
        job = self.backend.run(qc.final_circuit, shots=self.shots, noise_model=self.noise_model)
      else:
        job = self.backend.run(qc.final_circuit, shots=self.shots)
      counts = job.result().get_counts()

      statistics = {k: {d : 0 for d in ['0','1']} for k in range(self.n)}
      for output, count in counts.items():
        for ix, d in enumerate(reversed(output)):
          statistics[ix][d] += count

      for i in range(self.n):
        evolution[t,i] = statistics[i]['1']/self.shots

    return evolution