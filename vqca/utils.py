import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from scipy.stats import entropy

def haar_expressivity(ansatz : QuantumCircuit, **kwargs): 
    """
    Haar Metric for expressivity of VQA ansatz circuits.

    Args:
        ansatz (qiskit.QuantumCircuit):
        num_samples (int): 
        shots (int) : Default: 1024
        bins (int) : Default: 75
        random_seed 9int) : Default : 0
        noise (bool) : Default: False
    
    Returns:
        float: the haar metric
    """

    # Inicialization
    num_params = kwargs['num_parameters']
    num_samples = kwargs.get('num_samples', 1000)
    random_seed = kwargs.get('random_seed', 0)
    num_qubits = kwargs['num_qubits']
    shots = kwargs.get('shots', 50)
    bins = kwargs.get('bins', 100)
    
    simulator = AerSimulator()
    samples_fidelity = []
    circuits = []
    
    np.random.seed(random_seed)

    print(ansatz.draw())
    
    # Compute-Uncompute Fidelity
    for _ in range(num_samples):
        theta1 = np.random.uniform(0, 2*np.pi, num_params)
        theta2 = np.random.uniform(0, 2*np.pi, num_params)
        
        qc = QuantumCircuit(num_qubits)
        
        # Compute: U(θ₁) - Forward
        qc.compose(ansatz.assign_parameters(theta1), inplace=True)
        
        # Uncompute: U†(θ₂) - Backward
        ansatz_inverse = ansatz.assign_parameters(theta2).inverse()
        qc.compose(ansatz_inverse, inplace=True)
        
        qc.measure([i for i in range(num_qubits // 2)], [i for i in range(num_qubits // 2)])
        circuits.append(qc)
    
    # Execution and Probability Measurement
    circuits_transpiled = transpile(circuits, simulator)
    results = simulator.run(circuits_transpiled, shots=shots).result()
    
    state_zero = '0' * (num_qubits // 2)
    for i in range(num_samples):
        counts = results.get_counts(i)
        # |0...0> Probability
        state_zero_counts = counts.get(state_zero, 0)
        fidelity = state_zero_counts / shots
        samples_fidelity.append(fidelity)
        
    # KL Divergence 
    
    # Empirical P_PQC
    P_PQC, bins_buckets = np.histogram(samples_fidelity, bins=bins, range=(0, 1), density=True)
    P_PQC = P_PQC / np.sum(P_PQC) # Discrete probability normalization
    
    # Empirical P_Haar distribution
    N = 2 ** (qc.num_qubits * 2)
    bins_centers = (bins_buckets[:-1] + bins_buckets[1:]) / 2
    
    # P_Haar(F) = (N-1)(1-F)^(N-2)
    P_Haar = (N - 1.0) * (1.0 - bins_centers)**(N - 2)
    P_Haar = P_Haar / np.sum(P_Haar)
    
    # Zero smoothing to avoid log(0) in entropy calculation
    P_PQC = np.where(P_PQC == 0.0, 1e-10, P_PQC)
    P_Haar = np.where(P_Haar == 0.0, 1e-10, P_Haar)
    
    # KL Divergence: sum(P_PQC * log(P_PQC / P_Haar))
    kl_divergence = entropy(P_PQC, qk=P_Haar)
    
    return kl_divergence