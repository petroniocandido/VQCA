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
from qnca.qnca import QNCA
from qnca import ca_patterns 

class QNCAOptimizer(object):
  def __init__(self, **kwargs):
    self.hadamard = kwargs.get('hadamard',False)
    self.name = None
    self.min_loss = np.inf
    self.best_param = None
    self.loss_history = []
    self.param_history = []
    self.kwargs = kwargs
    self.pattern = kwargs['pattern']
    T,n = self.pattern.shape
    self.T = T
    self.n = n
    self.initial = self.pattern[0,:].tolist()
    self.dispositivo = 'GPU' if torch.cuda.is_available() else 'CPU'
    self.operator = kwargs['operator']
    self.num_param = int(str(self.operator)[:-1])
    self.shots = kwargs.get('shots',33)

    if 'backend' in kwargs:
      self.backend = kwargs['backend']
    else:
      self.backend = Aer.get_backend('qasm_simulator', device=self.dispositivo)

  def clean(self):
    self.loss_history = []
    self.param_history = []
    self.min_loss = np.inf
    self.best_param = None

  def log(self, loss, param):
    self.loss_history.append(loss)
    self.param_history.append(param.tolist())
    if loss < self.min_loss:
      self.min_loss = loss
      self.best_param = param.tolist()
      print("NEW MIN: {}\t{}".format(self.min_loss, self.best_param))

  def output(self, parametros):
    evolution = np.zeros((self.T,self.n))

    for t in range(self.T-1):
      qc = QNCA(operator = self.operator, initial=self.initial, T = t+1, \
                backend = self.backend, parametros = parametros, hadamard = self.hadamard)

      job = self.backend.run(qc.final_circuit, shots=self.shots)
      counts = job.result().get_counts()

      statistics = {k: {d : 0 for d in ['0','1']} for k in range(self.n)}
      for output, count in counts.items():
        for ix, d in enumerate(reversed(output)):
          statistics[ix][d] += count

      for i in range(self.n):
        evolution[t,i] = np.sum([int(k) * v for k,v in statistics[i].items()])/self.shots

    return evolution


  def mse(self, pattern2):
    mse = 0.0

    for t in range(self.T-1):
      for i in range(self.n):
        mse += (self.pattern[t+1,i] - pattern2[t,i])**2

    return mse
  

  def funcao_custo(self, parametros):   

    out = self.output(parametros)
    error = self.mse(out)
    self.log(error, parametros)

    return error



class QNCAGlobalOptimizer(object):
  def __init__(self, patterns, optimizer, **kwargs):
    self.patterns = patterns
    self.optimizer = optimizer
    self.kwargs = kwargs
    self.path = kwargs.get('path', '')
    self.hadamard = kwargs.get('hadamard',False)
    h = "H" if self.hadamard else ""
    self.file_path = self.path + "{}{}.json".format(h,optimizer.name)    
    self.finetunning_file_path = self.path + "{}{}-finetunning.json".format(h, optimizer.name)
    self.resume = kwargs.get('resume',True)
    
    self.history = {}
    if self.resume:
      if os.path.exists(self.file_path):
        with open(self.file_path, 'r') as file:
          self.history = json.load(file)

    self.finetunning = {}
    if os.path.exists(self.finetunning_file_path):
        with open(self.finetunning_file_path, 'r') as file:
          self.finetunning = json.load(file)
      

  def grid_search(self):
    for rule, pattern in self.patterns.items():
      print("\nRULE {}\n".format(rule))
      krule = str(rule)
      if not krule in self.history:
        self.history[krule] = {}
      for operator in [k for k in QNCA.operators if k < 30]:
        koperator = str(operator)
        if self.resume and koperator in self.history[krule]:
          continue
        print("\tOPERATOR {}\n".format(operator))
        self.history[krule][koperator] = {}

        optm = self.optimizer(pattern = np.array(pattern), operator = operator, **self.kwargs)
        optm.training_loop()

        self.history[krule][koperator]['loss_history'] = optm.loss_history
        self.history[krule][koperator]['param_history'] = optm.param_history                

        with open(self.file_path, "w") as outfile:
          json.dump(self.history, outfile)

  def global_training(self):

    for rule, pattern in self.patterns.items():
      print("\nRULE {}\n".format(rule))
      krule = str(rule)
      if not krule in self.history:
        self.history[krule] = {}
      for operator in QNCA.operators:
        koperator = str(operator)
        if self.resume and koperator in self.history[krule]:
          continue
        print("\tOPERATOR {}\n".format(operator))
        self.history[krule][koperator] = {}

        optm = self.optimizer(pattern = np.array(pattern), operator = operator, **self.kwargs)
        optm.training_loop()

        self.history[krule][koperator]['loss_history'] = optm.loss_history
        self.history[krule][koperator]['param_history'] = optm.param_history
        self.history[krule][koperator]['min_loss'] = optm.min_loss
        self.history[krule][koperator]['best_param'] = optm.best_param

        with open(self.file_path, "w") as outfile:
          json.dump(self.history, outfile)

  def k_perturbed_best(self, rule, operator, k, radius = 0.1):
    bp = self.history[rule][operator]['best_param']
    if k == 1:
      return bp 
    else:
      m = len(bp)
      tmp = ((np.random.randn(k-1,m) * radius) + bp).tolist()
      tmp.insert(0,bp)
    
      return tmp


  def fine_tunning(self, k = 1):    

    for rule, pattern in self.patterns.items():
      print("\nRULE {}\n".format(rule))
      krule = str(rule)
      if not krule in self.finetunning:
        self.finetunning[krule] = {}
      for operator in QNCA.operators:
        koperator = str(operator)
        if self.resume and koperator in self.finetunning[krule]:
          continue
        print("\tOPERATOR {}\n".format(operator))
        self.finetunning[krule][koperator] = {}

        optm = self.optimizer(pattern = np.array(pattern), operator = operator, **self.kwargs)
        best_perturbed = self.k_perturbed_best(krule, koperator, k)
        optm.training_loop(param = best_perturbed)

        self.finetunning[krule][koperator]['loss_history'] = optm.loss_history
        self.finetunning[krule][koperator]['param_history'] = optm.param_history
        self.finetunning[krule][koperator]['min_loss'] = optm.min_loss
        self.finetunning[krule][koperator]['best_param'] = optm.best_param

        with open(self.finetunning_file_path, "w") as outfile:
          json.dump(self.finetunning, outfile)


  def parse_dataframe(self):
    rows = []
    for rule in self.history.keys():
      for operator in self.history[rule].keys():
        row = [rule, operator, self.history[rule][operator]['min_loss']]
        rows.append(row)
    
    return pd.DataFrame(rows, columns=['Rule','Operator','MinLoss'])
  
  def parse_dataframe_ft(self):
    rows = []
    for rule in self.finetunning.keys():
      for operator in self.finetunning[rule].keys():
        row = [rule, operator, self.finetunning[rule][operator]['min_loss']]
        rows.append(row)
    
    return pd.DataFrame(rows, columns=['Rule','Operator','MinLoss'])

  def plot_results(self, **kwargs):
    rule = kwargs.get('rule',None)
    operator = kwargs.get('operator',None)

    num_results = 2 if len(self.finetunning) > 0 else 1

    fig, ax = plt.subplots(1, num_results, figsize=(10 * num_results, 10 ))

    if rule is not None and operator is not None:
      if num_results == 1:
        ax.plot(self.history[rule][operator]['loss_history'])
      else:
        ax[0].plot(self.history[rule][operator]['loss_history'])
        ax[1].plot(self.finetunning[rule][operator]['loss_history'])
    elif rule is not None and operator is None:

      df = self.parse_dataframe()
      df = df[df['Rule'] == rule]
      operators = df['Operator'].unique().tolist()
      df['X'] = [operators.index(k) for k in df['Operator'].values]
      
      if num_results == 1:        
        ax.bar(df['X'].values, df['MinLoss'].values)
        ax.set_xticks([k for k in range(len(operators))], operators)
      else:
        ax[0].bar(df['X'].values, df['MinLoss'].values)
        ax[0].set_xticks([k for k in range(len(operators))], operators)

        df = self.parse_dataframe_ft()
        df = df[df['Rule'] == rule]
        operators = df['Operator'].unique().tolist()
        df['X'] = [operators.index(k) for k in df['Operator'].values]

        ax[1].bar(df['X'].values, df['MinLoss'].values)
        ax[1].set_xticks([k for k in range(len(operators))], operators)
    else:
      df = self.parse_dataframe()
      rules = df['Rule'].unique().tolist()
      operators = df['Operator'].unique().tolist()

      df['X'] = [rules.index(k) for k in df['Rule'].values]
      df['Y'] = [operators.index(k) for k in df['Operator'].values]
      cmin = df['MinLoss'].min()
      cmax = df['MinLoss'].max()
      df['C'] = (df['MinLoss'].values - cmin) / (cmax - cmin)

      if num_results == 1: 
        ax.scatter(df['X'].values, df['Y'].values, s=df['MinLoss'].values * 50, c = df['MinLoss'].values, cmap='viridis' )
        ax.set_xticks([k for k in range(len(rules))], rules)
        ax.set_yticks([k for k in range(len(operators))], operators)
      else:
        ax[0].scatter(df['X'].values, df['Y'].values, s=df['MinLoss'].values * 50, c = df['MinLoss'].values, cmap='viridis' )
        ax[0].set_xticks([k for k in range(len(rules))], rules)
        ax[0].set_yticks([k for k in range(len(operators))], operators)

        df = self.parse_dataframe_ft()
        rules = df['Rule'].unique().tolist()
        operators = df['Operator'].unique().tolist()

        df['X'] = [rules.index(k) for k in df['Rule'].values]
        df['Y'] = [operators.index(k) for k in df['Operator'].values]
        cmin = df['MinLoss'].min()
        cmax = df['MinLoss'].max()
        df['C'] = (df['MinLoss'].values - cmin) / (cmax - cmin)

        ax[1].scatter(df['X'].values, df['Y'].values, s=df['MinLoss'].values * 50, c = df['MinLoss'].values, cmap='viridis' )
        ax[1].set_xticks([k for k in range(len(rules))], rules)
        ax[1].set_yticks([k for k in range(len(operators))], operators)


        #plt.colorbar()#boundaries=(cmin, cmax))
  
    plt.tight_layout()


  def sample(self, pattern, operator, params):
    shots = 33
    T,n = pattern.shape

    initial = pattern[0,:].tolist()

    evolution = np.zeros((T,n))

    for i in range(n):
      evolution[0,i] = initial[i]

    backend = Aer.get_backend('qasm_simulator', device="GPU")
    for t in range(T-1):
      qc = QNCA(operator = operator, initial=initial, T = t+1, backend = backend, parametros = params)

      job = backend.run(qc.final_circuit, shots=shots)
      counts = job.result().get_counts()

      statistics = {k: {d : 0 for d in ['0','1']} for k in range(n)}
      for output, count in counts.items():
        for ix, d in enumerate(reversed(output)):
          statistics[ix][d] += count

      for i in range(n):
        evolution[t+1,i] = np.sum([int(k) * v for k,v in statistics[i].items()])/shots

    return evolution

  def _plot_outputs_best(self, rule, operator, finetunning = False):
    if finetunning:
      return self.finetunning[rule][operator]['best_param']
    else:
      return self.history[rule][operator]['best_param']

  def plot_outputs(self, **kwargs):
    rule = kwargs.get('rule',None)
    operator = kwargs.get('operator',None)

    ft = kwargs.get('finetunning', False)

    if rule is not None and operator is not None:

      r = ca_patterns.rules[int(rule)]

      param = self._plot_outputs_best(rule, operator, ft)
      evol = self.sample(np.array(r), int(operator), np.array(param))

      fig, ax = plt.subplots(1,2)
      ax[0].matshow(r, cmap='Greys')
      ax[1].matshow(evol, cmap='Greys')

    elif rule is not None:
      r = ca_patterns.rules[int(rule)]
      no = len(QNCA.operators)
      fig, ax = plt.subplots(1,no+1, figsize=(20, 5))
      ax[0].matshow(r, cmap='Greys')
      ax[0].set_title("Pattern {}".format(rule))
      for ct, op in enumerate(QNCA.operators):
        #print(ct, op)
        self._plot_outputs_axis(ax[ct+1], rule, r, op, ft)

    else:
      nr = len(ca_patterns.rules)
      no = len(QNCA.operators)
      fig, ax = plt.subplots(nr,no+1, figsize=(20, 2*nr))

      for ct1, rule in enumerate(ca_patterns.rules.keys()):
        r = ca_patterns.rules[rule]
        ax[ct1, 0].matshow(r, cmap='Greys')
        ax[ct1, 0].set_title("Pattern {}".format(rule))
        for ct2, op in enumerate(QNCA.operators):
          self._plot_outputs_axis(ax[ct1, ct2+1], str(rule), r, op, ft)
   
    plt.tight_layout()

  def _plot_outputs_axis(self, ax, rule, pattern, op, ft):
    #param = self.history[rule][str(op)]['best_param']
    param = self._plot_outputs_best(rule, str(op), ft)
    evol = self.sample(np.array(pattern), op, np.array(param))
    ax.matshow(evol, cmap='Greys')
    ax.set_title(str(op))
    ax.set_xlabel(str(round(self.history[rule][str(op)]['min_loss'],3)))
    ax.set_xticks([])
    ax.set_yticks([])

  def plot(self):
    plt.plot(self.loss_history)




