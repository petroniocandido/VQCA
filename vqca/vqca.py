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
  """Variational Quantum Cellular Automata instance.

   VQCA is a Qiskit parameterized quantum circuit representing an entire quantum cellular automata simulation 
   with `T` iterations starting from an `initial` configuration and using a specific `operator`circuit for
   cell state transition.

    Attributes:
        n (int): Number of cellular automata cells
        T (int): Number of simulation iterations
        initial (string): Initial state of the cellular automada
        operator (VQCAOperator): Cell state transition circuit
        hadamard (bool): If the hadamard transform should be applied in each cell state transtition circuit
        iqp (bool): If an iqp preparing should be applied in each cell state transtition circuit
        reset (bool): If the reset gate should be applied at the end of each iteration.  Attention! This parameter makes the VQCA an irreversible circuit
        backend (qiskit.Backend): The Qiskit execution backend.
        parameters (numpy.ndarray) : Assign the array of parameters
        full_reversible (bool) : Don't include irreversible gates (measurement and reset) in the circuit. Default: False
    """

  def __init__(self,**kwargs):
    """
    Build an VQCA circuit instance.

    Args:
      n (int): Number of cellular automata cells
      T (int): Number of simulation iterations
      initial (string): Initial state of the cellular automada
      operator (VQCAOperator): Cell state transition circuit
      hadamard (bool): If the hadamard transform should be applied in each cell state transtition circuit
      iqp (bool): If an iqp preparing should be applied in each cell state transtition circuit
      reset (bool): If the reset gate should be applied at the end of each iteration.  Attention! This parameter makes the VQCA an irreversible circuit
      backend (qiskit.Backend): The Qiskit execution backend.
      parameters (numpy.ndarray) : Assign the array of parameters
      full_reversible (bool) : Don't include irreversible gates (measurement and reset) in the circuit. Default: False
    """

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
    self.reset = kwargs.get('reset',False)   # Attention! This parameter makes the VQCA an irreversible circuit
    self.full_reversible = kwargs.get('full_reversible',False)   
    if self.full_reversible:
      self.reset = False

    self.ix = list(range(self.n))
    self.backend = kwargs.get('backend', None)
    self.compiled_circuit = None
    self.final_circuit = None

    self.build_circuit()
    self.transpile()

    if 'parameters' in kwargs:
      self.assign_parameters(kwargs['parameters'])

  ###
  # |ψ⟩^0 = S
  ###
  def init(self):
    """
    Initial circuit configuration, from |0⟩^2n to initial state |ψ⟩ = S
    """
    for i in range(self.n):
      if self.initial[i]:
        self.qc.x(i)

  def exec_hadamard(self):
    """
    Configure the circuit for two special parameters: hadamard and IQP
    """
    if self.iqp:
      self.qc.h([k for k in range(2*self.n)])
    elif self.hadamard:
        self.qc.h([k for k in range(self.n, 2*self.n)])
  
  def build_circuit(self):
    """
    Generate the logical circuit, applying all parameters, initial state and iterated transition operators.
    This is the first phase, followed by transpilation.

    """

    # Apply initial state
    self.init()

    for t in range(self.T):
      # Apply initial hadamard transform or prepatre IQP circuit, if they are flagged
      self.exec_hadamard()

      # Apply cell state trasition circuits
      self.qc = self.operator.apply(t, self.qc)

      # Apply final hadamard transform or prepatre IQP circuit, if they are flagged
      self.exec_hadamard()

      for i in self.ix:

        # Swap ancilla and cell states
        #|ψ⟩^t <-- |ψ⟩^t+1
        self.qc.swap(i+self.n, i)

        # Attention! This makes the VQCA an irreversible circuit
        # Apply reset gates in ancillas, if flagged
        #|ψ⟩^t+1 <-- |0⟩
        if self.reset:
          self.qc.reset(i+self.n) 

    # Avoid measurement gates, for fidelity calculations
    if not self.full_reversible:
      self.qc.measure(self.ix, self.ix)

  def transpile(self):
    """
    Transpilation of logical circuit in a physical circuit of the given backend.
    This is the second phase, preceded by build_circuit and followed by assing_parameters.
    """
    self.compiled_circuit = transpile(self.qc, self.backend)

  def assign_parameters(self, parameters):
    """
    Prepare the final circuit assigning the PQC parameters.
    This is the third and final phase, preceded by transpile.

    Args:
      parameters (numpy.ndarray) : Arry of parameters values 
    """
    self.final_circuit = self.compiled_circuit.assign_parameters({
      self.operator.p : parameters
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
