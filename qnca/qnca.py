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

class QNCA(object):

  operators = [21, 22, 30, 31, 32, 33, 41, 42, 43, 60, 91, 92, 120, 150, 180]

  def __init__(self,**kwargs):
    self.n = kwargs.get('n',0)
    self.initial = kwargs.get('initial',None)
    self.hadamard = kwargs.get('hadamard',False)
    if self.n == 0 and self.initial is not None:
      self.n = len(self.initial)
    elif self.n > 0 and self.initial is None:
      self.initial = [0 for k in range(self.n)]
    elif self.n == 0 and self.initial is not None:
      raise Exception("Or n or initial should be informed!")
    self.T = kwargs.get('T', 3)
    self.qc = QuantumCircuit(2*self.n, self.n)
    self.operator = kwargs.get('operator', 18)
    self.p = ParameterVector('ϴ', int(str(self.operator)[:-1]))
    self.ix = list(range(self.n))
    self.backend = kwargs.get('backend', 18)
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
    if self.hadamard:
        self.qc.h([k for k in range(self.n, 2*self.n)])


  ###
  # |ψ⟩^t+1 <-- U|ψ⟩^t
  ###

  def unitary_operator_21(self):
    for i in self.ix:
      
      self.qc.cx(i, i+self.n)
      self.qc.crx(self.p[0], (i+1)%self.n, i+self.n)
      self.qc.crx(self.p[1], (i-1)%self.n, i+self.n)

  def unitary_operator_22(self):
    for i in self.ix:
      self.qc.cx(i, i+self.n)
      self.qc.cry(self.p[0], (i+1)%self.n, i+self.n)
      self.qc.cry(self.p[1], (i-1)%self.n, i+self.n)

  def unitary_operator_30(self):
    for i in self.ix:
      self.qc.cx(i, i+self.n)
      self.qc.crx(self.p[0], (i+1)%self.n, i+self.n)
      self.qc.crx(self.p[1], (i-1)%self.n, i+self.n)
      self.qc.rx(self.p[2], i+self.n)

  def unitary_operator_31(self):
    for i in self.ix:
      self.qc.cx(i, i+self.n)
      self.qc.cry(self.p[0], (i+1)%self.n, i+self.n)
      self.qc.cry(self.p[1], (i-1)%self.n, i+self.n)
      self.qc.ry(self.p[2], i+self.n)

  def unitary_operator_32(self):
    for i in self.ix:
      self.qc.crx(self.p[0], i, i+self.n)
      self.qc.crx(self.p[1], (i+1)%self.n, i+self.n)
      self.qc.crx(self.p[2], (i-1)%self.n, i+self.n)

  def unitary_operator_33(self):
    for i in self.ix:
      self.qc.cry(self.p[0], i, i+self.n)
      self.qc.cry(self.p[1], (i+1)%self.n, i+self.n)
      self.qc.cry(self.p[2], (i-1)%self.n, i+self.n)

  def unitary_operator_41(self):
    for i in self.ix:
      self.qc.crx(self.p[0], i, i+self.n)
      self.qc.crx(self.p[1], (i+1)%self.n, i+self.n)
      self.qc.crx(self.p[2], (i-1)%self.n, i+self.n)
      self.qc.rx(self.p[3], i+self.n)

  def unitary_operator_42(self):
    for i in self.ix:
      self.qc.crx(self.p[0], i, i+self.n)
      self.qc.crx(self.p[1], (i+1)%self.n, i+self.n)
      self.qc.crx(self.p[2], (i-1)%self.n, i+self.n)
      self.qc.ry(self.p[3], i+self.n)

  def unitary_operator_43(self):
    for i in self.ix:
      self.qc.cry(self.p[0], i, i+self.n)
      self.qc.cry(self.p[1], (i+1)%self.n, i+self.n)
      self.qc.cry(self.p[2], (i-1)%self.n, i+self.n)
      self.qc.ry(self.p[3], i+self.n)

  def unitary_operator_60(self):
    for i in self.ix:
      self.qc.cx(i, i+self.n)
      self.qc.cu(self.p[0], self.p[1], self.p[2], 0, (i+1)%self.n, i+self.n)
      self.qc.cu(self.p[3], self.p[4], self.p[5], 0, (i-1)%self.n, i+self.n)


  def unitary_operator_91(self):
    for i in self.ix:
      self.qc.cx(i, i+self.n)
      self.qc.cu(self.p[0], self.p[1], self.p[2], 0, (i+1)%self.n, i+self.n)
      self.qc.cu(self.p[3], self.p[4], self.p[5], 0, (i-1)%self.n, i+self.n)
      self.qc.u(self.p[6], self.p[7], self.p[8], i+self.n)

  def unitary_operator_92(self):
    for i in self.ix:
      self.qc.cu(self.p[0], self.p[1], self.p[2], 0, i, i+self.n)
      self.qc.cu(self.p[3], self.p[4], self.p[5], 0, (i+1)%self.n, i+self.n)
      self.qc.cu(self.p[6], self.p[7], self.p[8], 0, (i-1)%self.n, i+self.n)

  def unitary_operator_120(self):
    for i in self.ix:
      self.qc.cu(self.p[0], self.p[1], self.p[2], 0, i, i+self.n)
      self.qc.cu(self.p[3], self.p[4], self.p[5], 0, (i+1)%self.n, i+self.n)
      self.qc.cu(self.p[6], self.p[7], self.p[8], 0, (i-1)%self.n, i+self.n)
      self.qc.u(self.p[9], self.p[10], self.p[11], i+self.n)

  def unitary_operator_150(self):
    for i in self.ix:
      self.qc.cx(i, i+self.n)
      self.qc.cu(self.p[0], self.p[1], self.p[2], 0, (i+1)%self.n, i+self.n)
      self.qc.cu(self.p[3], self.p[4], self.p[5], 0, (i-1)%self.n, i+self.n)
      self.qc.u(self.p[6], self.p[7], self.p[8], i+self.n)
      self.qc.cu(self.p[9], self.p[10], self.p[11], 0, (i+1)%self.n, i+self.n)
      self.qc.cu(self.p[12], self.p[13], self.p[14], 0, (i-1)%self.n, i+self.n)

  def unitary_operator_180(self):
    for i in self.ix:
      self.qc.cu(self.p[0], self.p[1], self.p[2], 0, i, i+self.n)
      self.qc.cu(self.p[3], self.p[4], self.p[5], 0, (i+1)%self.n, i+self.n)
      self.qc.cu(self.p[6], self.p[7], self.p[8], 0, (i-1)%self.n, i+self.n)
      self.qc.u(self.p[9], self.p[10], self.p[11], i+self.n)
      self.qc.cu(self.p[12], self.p[13], self.p[14], 0, (i+1)%self.n, i+self.n)
      self.qc.cu(self.p[15], self.p[16], self.p[17], 0, (i-1)%self.n, i+self.n)

  def build_circuit(self):
    self.init()

    for t in range(self.T):
      self.exec_hadamard()

      match self.operator:
        case 21:
          self.unitary_operator_21()

        case 22:
          self.unitary_operator_22()

        case 30:
          self.unitary_operator_30()

        case 31:
          self.unitary_operator_31()

        case 32:
          self.unitary_operator_32()

        case 33:
          self.unitary_operator_33()

        case 41:
          self.unitary_operator_41()

        case 42:
          self.unitary_operator_42()

        case 43:
          self.unitary_operator_43()

        case 60:
          self.unitary_operator_60()

        case 91:
          self.unitary_operator_91()

        case 92:
          self.unitary_operator_92()

        case 120:
          self.unitary_operator_120()

        case 150:
          self.unitary_operator_150()

        case 180:
          self.unitary_operator_180()      

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
      self.p : parametros
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
