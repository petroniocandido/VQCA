from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer import  Aer, AerSimulator
from qiskit import transpile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import json
import os
from vqca.operators import get_id, VQCAOperator


class VQCA(object):

  def __init__(self,**kwargs):
    self.n = kwargs.get('n',0)
    self.initial = kwargs.get('initial',None)    
    
    if self.n == 0 and self.initial is not None:
      self.n = len(self.initial)
    elif self.n > 0 and self.initial is None:
      self.initial = [0 for k in range(self.n)]
    elif self.n == 0 and self.initial is not None:
      raise Exception("Or n or initial state should be informed!")
    
    op = kwargs.get('operator', None)

    if op is None:
      raise Exception("An operator id or instance must be informed!")
    elif isinstance(op, int):
      self.operator = get_id(self.n, op)
    elif isinstance(op, VQCAOperator):
      self.operator = op

    self.T = kwargs.get('T', 3)
    self.qc = QuantumCircuit(2*self.n, self.n)
    self.hadamard = kwargs.get('hadamard',False)
    self.iqp = kwargs.get('iqp',False)

    self.ix = list(range(self.n))
    self.backend = kwargs.get('backend', None)
    self.compiled_circuit = None
    self.final_circuit = None

    self.build_circuit()
    self.transpile()

    if 'parametros' in kwargs:
      self.assign_parameters(kwargs['parametros'])

  ###
  # |ψ⟩^0 = S
  ###
  def init(self):
    for i in range(self.n):
      if self.initial[i]:
        self.qc.x(i)

  def exec_hadamard(self):
    if self.iqp:
      self.qc.h([k for k in range(2*self.n)])
    elif self.hadamard:
        self.qc.h([k for k in range(self.n, 2*self.n)])
  
  def build_circuit(self):
    self.init()

    for t in range(self.T):
      self.exec_hadamard()

      self.qc = self.operator.apply(t, self.qc)

      self.exec_hadamard()

      for i in self.ix:

        #|ψ⟩^t <-- |ψ⟩^t+1
        self.qc.swap(i+self.n, i)

        #|ψ⟩^t+1 <-- |0⟩
        self.qc.reset(i+self.n)

    self.qc.measure(self.ix, self.ix)

  def transpile(self):
    self.compiled_circuit = transpile(self.qc, self.backend)


  def assign_parameters(self, parametros):
    self.final_circuit = self.compiled_circuit.assign_parameters({
      self.operator.p : parametros
    })

  def logical_circuit(self):
    return self.qc

  def logical_circuit_shape(self):
    return (self.qc.num_qubits, self.qc.depth(), self.qc.size())

  def physical_circuit(self):
    return self.compiled_circuit

  def physical_circuit_shape(self):
    return (self.qc.num_qubits, self.qc.depth(), self.qc.size())

  def final_circuit(self):
    return self.final_circuit
