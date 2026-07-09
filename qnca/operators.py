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


class VQCAOperator(object):
    def __init__(self,**kwargs):
        self.n = kwargs.get("n", 1)
        self.npar = kwargs.get("npar", 1)
        self.id = kwargs.get("id", 1)
        self.p = ParameterVector('ϴ', self.npar)

    def apply(self, qc):
        for ix in list(range(self.n)):
            self.circuit(qc)
    
    def circuit(self, i, qc):
        pass


class Operator21(VQCAOperator):
    def __init__(self,**kwargs):
        super(Operator21, self).__init__(npar = 2, id = 1, **kwargs)
    
    def circuit(self, i, qc):
        qc.cx(i, i+self.n)
        qc.crx(self.p[0], (i+1)%self.n, i+self.n)
        qc.crx(self.p[1], (i-1)%self.n, i+self.n)


class Operator22(VQCAOperator):
    def __init__(self,**kwargs):
        super(Operator22, self).__init__(npar = 2, id = 1, **kwargs)
    
    def circuit(self, i, qc):
        qc.cx(i, i+self.n)
        qc.cry(self.p[0], (i+1)%self.n, i+self.n)
        qc.cry(self.p[1], (i-1)%self.n, i+self.n)
        

class Operator30(VQCAOperator):
    def __init__(self,**kwargs):
        super(Operator30, self).__init__(npar = 3, id = 0, **kwargs)
    
    def circuit(self, i, qc):
        qc.cx(i, i+self.n)
        qc.crx(self.p[0], (i+1)%self.n, i+self.n)
        qc.crx(self.p[1], (i-1)%self.n, i+self.n)
        qc.rx(self.p[2], i+self.n)


class Operator31(VQCAOperator):
    def __init__(self,**kwargs):
        super(Operator31, self).__init__(npar = 3, id = 1, **kwargs)
    
    def circuit(self, i, qc):
        qc.cx(i, i+self.n)
        qc.cry(self.p[0], (i+1)%self.n, i+self.n)
        qc.cry(self.p[1], (i-1)%self.n, i+self.n)
        qc.ry(self.p[2], i+self.n)


class Operator32(VQCAOperator):
    def __init__(self,**kwargs):
        super(Operator32, self).__init__(npar = 3, id = 2, **kwargs)
    
    def circuit(self, i, qc):
        qc.crx(self.p[0], i, i+self.n)
        qc.crx(self.p[1], (i+1)%self.n, i+self.n)
        qc.crx(self.p[2], (i-1)%self.n, i+self.n)


class Operator33(VQCAOperator):
    def __init__(self,**kwargs):
        super(Operator33, self).__init__(npar = 3, id = 3, **kwargs)
    
    def circuit(self, i, qc):
        qc.cry(self.p[0], i, i+self.n)
        qc.cry(self.p[1], (i+1)%self.n, i+self.n)
        qc.cry(self.p[2], (i-1)%self.n, i+self.n)


class Operator41(VQCAOperator):
    def __init__(self,**kwargs):
        super(Operator41, self).__init__(npar = 4, id = 1, **kwargs)
    
    def circuit(self, i, qc):
        qc.crx(self.p[0], i, i+self.n)
        qc.crx(self.p[1], (i+1)%self.n, i+self.n)
        qc.crx(self.p[2], (i-1)%self.n, i+self.n)
        qc.rx(self.p[3], i+self.n)


class Operator42(VQCAOperator):
    def __init__(self,**kwargs):
        super(Operator42, self).__init__(npar = 4, id = 2, **kwargs)
    
    def circuit(self, i, qc):
        qc.crx(self.p[0], i, i+self.n)
        qc.crx(self.p[1], (i+1)%self.n, i+self.n)
        qc.crx(self.p[2], (i-1)%self.n, i+self.n)
        qc.ry(self.p[3], i+self.n)


class Operator43(VQCAOperator):
    def __init__(self,**kwargs):
        super(Operator43, self).__init__(npar = 4, id = 3, **kwargs)
    
    def circuit(self, i, qc):
        qc.cry(self.p[0], i, i+self.n)
        qc.cry(self.p[1], (i+1)%self.n, i+self.n)
        qc.cry(self.p[2], (i-1)%self.n, i+self.n)
        qc.ry(self.p[3], i+self.n)


class Operator60(VQCAOperator):
    def __init__(self,**kwargs):
        super(Operator60, self).__init__(npar = 6, id = 0, **kwargs)
    
    def circuit(self, i, qc):
        qc.cx(i, i+self.n)
        qc.cu(self.p[0], self.p[1], self.p[2], 0, (i+1)%self.n, i+self.n)
        qc.cu(self.p[3], self.p[4], self.p[5], 0, (i-1)%self.n, i+self.n)


class Operator91(VQCAOperator):
    def __init__(self,**kwargs):
        super(Operator91, self).__init__(npar = 9, id = 1, **kwargs)
    
    def circuit(self, i, qc):
        qc.cx(i, i+self.n)
        qc.cu(self.p[0], self.p[1], self.p[2], 0, (i+1)%self.n, i+self.n)
        qc.cu(self.p[3], self.p[4], self.p[5], 0, (i-1)%self.n, i+self.n)
        qc.u(self.p[6], self.p[7], self.p[8], i+self.n)


class Operator92(VQCAOperator):
    def __init__(self,**kwargs):
        super(Operator92, self).__init__(npar = 9, id = 2, **kwargs)
    
    def circuit(self, i, qc):
        qc.cu(self.p[0], self.p[1], self.p[2], 0, i, i+self.n)
        qc.cu(self.p[3], self.p[4], self.p[5], 0, (i+1)%self.n, i+self.n)
        qc.cu(self.p[6], self.p[7], self.p[8], 0, (i-1)%self.n, i+self.n)


class Operator120(VQCAOperator):
    def __init__(self,**kwargs):
        super(Operator120, self).__init__(npar = 12, id = 0, **kwargs)
    
    def circuit(self, i, qc):
        qc.cu(self.p[0], self.p[1], self.p[2], 0, i, i+self.n)
        qc.cu(self.p[3], self.p[4], self.p[5], 0, (i+1)%self.n, i+self.n)
        qc.cu(self.p[6], self.p[7], self.p[8], 0, (i-1)%self.n, i+self.n)
        qc.u(self.p[9], self.p[10], self.p[11], i+self.n)


class Operator150(VQCAOperator):
    def __init__(self,**kwargs):
        super(Operator150, self).__init__(npar = 15, id = 0, **kwargs)
    
    def circuit(self, i, qc):
        qc.cx(i, i+self.n)
        qc.cu(self.p[0], self.p[1], self.p[2], 0, (i+1)%self.n, i+self.n)
        qc.cu(self.p[3], self.p[4], self.p[5], 0, (i-1)%self.n, i+self.n)
        qc.u(self.p[6], self.p[7], self.p[8], i+self.n)
        qc.cu(self.p[9], self.p[10], self.p[11], 0, (i+1)%self.n, i+self.n)
        qc.cu(self.p[12], self.p[13], self.p[14], 0, (i-1)%self.n, i+self.n)


class Operator180(VQCAOperator):
    def __init__(self,**kwargs):
        super(Operator180, self).__init__(npar = 18, id = 0, **kwargs)
    
    def circuit(self, i, qc):
        qc.cu(self.p[0], self.p[1], self.p[2], 0, i, i+self.n)
        qc.cu(self.p[3], self.p[4], self.p[5], 0, (i+1)%self.n, i+self.n)
        qc.cu(self.p[6], self.p[7], self.p[8], 0, (i-1)%self.n, i+self.n)
        qc.u(self.p[9], self.p[10], self.p[11], i+self.n)
        qc.cu(self.p[12], self.p[13], self.p[14], 0, (i+1)%self.n, i+self.n)
        qc.cu(self.p[15], self.p[16], self.p[17], 0, (i-1)%self.n, i+self.n)
    

operators = {
    21 : Operator21, 22 : Operator22, 30 : Operator30, 31 : Operator31, 32 : Operator32, 33 : Operator33, 
    40 : Operator41, 41 : Operator42, 43 : Operator43, 60 : Operator60, 91 : Operator91, 92 : Operator92,
    120 : Operator120, 150 : Operator150, 180 : Operator180
}


def get_operator(n, npar, id):
    op = operators[npar * 10 + id]
    return op(n = n)


def get_id(n, oid):
    soid = str(oid)
    id = int(soid[-1])
    npar = int(soid[:-1])
    return get_operator(n, npar, id)

