from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer import  Aer, AerSimulator
from qiskit_aer.noise import  NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from qiskit import transpile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import json
import os
from vqca.vqca import VQCA
from vqca import ca_patterns 
from vqca.operators import VQCAOperator, get_id, operators
from vqca.estimator import VQCAEstimator
from vqca.utils import haar_expressivity

class VQCACircuitStatistics(VQCAEstimator):
  def __init__(self, **kwargs):
    super(VQCACircuitStatistics, self).__init__(**kwargs)

    self.path = kwargs.get('path', '')

    self.initial = ca_patterns.rule1[0]
      
    prefix = "N" if self.noise else ""
    if self.iqp:
      prefix += "I" 
    elif self.hadamard:
      prefix += "H"

    if self.reset:
      prefix = 'R' + prefix

    self.file_path = self.path + "{}statistics.json".format(prefix)    
    self.resume = kwargs.get('resume',True)
    
    self.statistics = {}
    if self.resume:
      if os.path.exists(self.file_path):
        with open(self.file_path, 'r') as file:
          self.statistics = json.load(file)

  def run(self):
    for id, operator in operators.items():
      print("Operator: {}".format(id))
      if not id in self.statistics:
        self.statistics[id] = {}
      else:
        continue

      qc = VQCA(n = self.n, operator = id, \
                initial= self.initial, T = self.T, \
                iqp = self.iqp, hadamard = self.hadamard,\
                backend = self.backend,  full_reversible = True)

      
      log_q, log_d, log_s = qc.logical_circuit_shape()
      phys_q, phys_d, phys_s = qc.physical_circuit_shape()
      self.statistics[id]["logical"] = {"qubits" : log_q, "depth": log_d, "size": log_s}
      self.statistics[id]["physical"] = {"qubits" : phys_q, "depth": phys_d, "size": phys_s}
      self.statistics[id]["haar"] = haar_expressivity(qc.logical_circuit(),
                                                      num_qubits = self.n*2,
                                                      num_parameters = self.num_param)
      with open(self.file_path, "w") as outfile:
        json.dump(self.statistics, outfile)

      
      
      
    