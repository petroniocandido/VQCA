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

    def apply(self, t, qc):
        for i in list(range(self.n)):
            qc = self.circuit(i, qc)
        return qc
    
    def circuit(self, i, qc):
        pass

    def __str__(self):
        print("{}-{}-{}".format(self.n, self.npar, self.id))


class Operator21(VQCAOperator):
    def __init__(self, **kwargs):
        super(Operator21, self).__init__(npar = 2, id = 1, **kwargs)
    
    def circuit(self, i, qc):
        qc.cx(i, i+self.n)
        qc.crx(self.p[0], (i+1)%self.n, i+self.n)
        qc.crx(self.p[1], (i-1)%self.n, i+self.n)
        return qc


class Operator22(VQCAOperator):
    def __init__(self,**kwargs):
        super(Operator22, self).__init__(npar = 2, id = 1, **kwargs)
    
    def circuit(self, i, qc):
        qc.cx(i, i+self.n)
        qc.cry(self.p[0], (i+1)%self.n, i+self.n)
        qc.cry(self.p[1], (i-1)%self.n, i+self.n)
        return qc
        

class Operator30(VQCAOperator):
    def __init__(self,**kwargs):
        super(Operator30, self).__init__(npar = 3, id = 0, **kwargs)
    
    def circuit(self, i, qc):
        qc.cx(i, i+self.n)
        qc.crx(self.p[0], (i+1)%self.n, i+self.n)
        qc.crx(self.p[1], (i-1)%self.n, i+self.n)
        qc.rx(self.p[2], i+self.n)
        return qc


class Operator31(VQCAOperator):
    def __init__(self,**kwargs):
        super(Operator31, self).__init__(npar = 3, id = 1, **kwargs)
    
    def circuit(self, i, qc):
        qc.cx(i, i+self.n)
        qc.cry(self.p[0], (i+1)%self.n, i+self.n)
        qc.cry(self.p[1], (i-1)%self.n, i+self.n)
        qc.ry(self.p[2], i+self.n)
        return qc


class Operator32(VQCAOperator):
    def __init__(self,**kwargs):
        super(Operator32, self).__init__(npar = 3, id = 2, **kwargs)
    
    def circuit(self, i, qc):
        qc.crx(self.p[0], i, i+self.n)
        qc.crx(self.p[1], (i+1)%self.n, i+self.n)
        qc.crx(self.p[2], (i-1)%self.n, i+self.n)
        return qc


class Operator33(VQCAOperator):
    def __init__(self,**kwargs):
        super(Operator33, self).__init__(npar = 3, id = 3, **kwargs)
    
    def circuit(self, i, qc):
        qc.cry(self.p[0], i, i+self.n)
        qc.cry(self.p[1], (i+1)%self.n, i+self.n)
        qc.cry(self.p[2], (i-1)%self.n, i+self.n)
        return qc


class Operator41(VQCAOperator):
    def __init__(self,**kwargs):
        super(Operator41, self).__init__(npar = 4, id = 1, **kwargs)
    
    def circuit(self, i, qc):
        qc.crx(self.p[0], i, i+self.n)
        qc.crx(self.p[1], (i+1)%self.n, i+self.n)
        qc.crx(self.p[2], (i-1)%self.n, i+self.n)
        qc.rx(self.p[3], i+self.n)
        return qc


class Operator42(VQCAOperator):
    def __init__(self,**kwargs):
        super(Operator42, self).__init__(npar = 4, id = 2, **kwargs)
    
    def circuit(self, i, qc):
        qc.crx(self.p[0], i, i+self.n)
        qc.crx(self.p[1], (i+1)%self.n, i+self.n)
        qc.crx(self.p[2], (i-1)%self.n, i+self.n)
        qc.ry(self.p[3], i+self.n)
        return qc


class Operator43(VQCAOperator):
    def __init__(self,**kwargs):
        super(Operator43, self).__init__(npar = 4, id = 3, **kwargs)
    
    def circuit(self, i, qc):
        qc.cry(self.p[0], i, i+self.n)
        qc.cry(self.p[1], (i+1)%self.n, i+self.n)
        qc.cry(self.p[2], (i-1)%self.n, i+self.n)
        qc.ry(self.p[3], i+self.n)
        return qc


class Operator60(VQCAOperator):
    def __init__(self,**kwargs):
        super(Operator60, self).__init__(npar = 6, id = 0, **kwargs)
    
    def circuit(self, i, qc):
        qc.cx(i, i+self.n)
        qc.cu(self.p[0], self.p[1], self.p[2], 0, (i+1)%self.n, i+self.n)
        qc.cu(self.p[3], self.p[4], self.p[5], 0, (i-1)%self.n, i+self.n)
        return qc


class Operator91(VQCAOperator):
    def __init__(self,**kwargs):
        super(Operator91, self).__init__(npar = 9, id = 1, **kwargs)
    
    def circuit(self, i, qc):
        qc.cx(i, i+self.n)
        qc.cu(self.p[0], self.p[1], self.p[2], 0, (i+1)%self.n, i+self.n)
        qc.cu(self.p[3], self.p[4], self.p[5], 0, (i-1)%self.n, i+self.n)
        qc.u(self.p[6], self.p[7], self.p[8], i+self.n)
        return qc


class Operator92(VQCAOperator):
    def __init__(self,**kwargs):
        super(Operator92, self).__init__(npar = 9, id = 2, **kwargs)
    
    def circuit(self, i, qc):
        qc.cu(self.p[0], self.p[1], self.p[2], 0, i, i+self.n)
        qc.cu(self.p[3], self.p[4], self.p[5], 0, (i+1)%self.n, i+self.n)
        qc.cu(self.p[6], self.p[7], self.p[8], 0, (i-1)%self.n, i+self.n)
        return qc


class Operator120(VQCAOperator):
    def __init__(self,**kwargs):
        super(Operator120, self).__init__(npar = 12, id = 0, **kwargs)
    
    def circuit(self, i, qc):
        qc.cu(self.p[0], self.p[1], self.p[2], 0, i, i+self.n)
        qc.cu(self.p[3], self.p[4], self.p[5], 0, (i+1)%self.n, i+self.n)
        qc.cu(self.p[6], self.p[7], self.p[8], 0, (i-1)%self.n, i+self.n)
        qc.u(self.p[9], self.p[10], self.p[11], i+self.n)
        return qc


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
        return qc


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
        return qc


class OperatorIQP6(VQCAOperator):
    def __init__(self,**kwargs):
        super(OperatorIQP6, self).__init__(npar = 6, id = 1, **kwargs)
    
    def circuit(self, i, qc):
        qc.ccz(i, (i+1)%self.n, i+self.n)
        qc.u(self.p[0], self.p[1], self.p[2], i+self.n)
        qc.ccz(i, (i-1)%self.n, i+self.n)
        qc.u(self.p[3], self.p[4], self.p[5], i+self.n)
        return qc
    
class OperatorIQP2(VQCAOperator):
    def __init__(self,**kwargs):
        super(OperatorIQP2, self).__init__(npar = 1, id = 3, **kwargs)
    
    def circuit(self, i, qc):
        qc.ccz(i, (i+1)%self.n, i+self.n)
        qc.rz(self.p[0], i+self.n)
        qc.ccz(i, (i-1)%self.n, i+self.n)
        #qc.rz(self.p[1], i+self.n)
        return qc
    

class OperatorIQP3(VQCAOperator):
    def __init__(self,**kwargs):
        super(OperatorIQP3, self).__init__(npar = 3, id = 4, **kwargs)
    
    def circuit(self, i, qc):
        qc.crz(self.p[0], i, i+self.n)
        qc.crz(self.p[1], (i+1)%self.n, i+self.n)
        qc.crz(self.p[2], (i-1)%self.n, i+self.n)        
        return qc
    


class OperatorIQP9(VQCAOperator):
    def __init__(self,**kwargs):
        super(OperatorIQP9, self).__init__(npar = 9, id = 3, **kwargs)
    
    def circuit(self, i, qc):
        qc.cz(i, i+self.n)
        qc.u(self.p[0], self.p[1], self.p[2], i+self.n)
        qc.cz((i+1)%self.n, i+self.n)
        qc.u(self.p[3], self.p[4], self.p[5], i+self.n)
        qc.cz((i-1)%self.n, i+self.n)
        qc.u(self.p[6], self.p[7], self.p[8], i+self.n)
        return qc



operators = {
    21 : Operator21, 22 : Operator22, 30 : Operator30, 31 : Operator31, 32 : Operator32, 33 : Operator33, 
    41 : Operator41, 42 : Operator42, 43 : Operator43, 60 : Operator60, 91 : Operator91, 92 : Operator92,
    120 : Operator120, 150 : Operator150, 180 : Operator180, 61 : OperatorIQP6, 11: OperatorIQP2, 34 : OperatorIQP3,
    93 : OperatorIQP9
}


def get_operator(n, npar, id):
    op = operators[npar * 10 + id]
    return op(n = n)


def get_id(n, oid):
    soid = str(oid)
    id = int(soid[-1])
    npar = int(soid[:-1])
    return get_operator(n, npar, id)

