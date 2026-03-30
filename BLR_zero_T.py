"""
Bayesian Linear Regression (BLR) with quadratic reward function.

Notation used consistently:
- n: int (CPU) number of training samples
- d: int (CPU) number of features
- n_test: int (CPU) number of test samples
- X: [n, d] or [n_test, d] on device
- w_T, w_R: [d] on device, teacher and reward weights
- mean/m: [B] on device (B is a batch size)
- std/s: [B] on device
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BayesianLinearRegression:
    def __init__(self, alpha, beta):
        """
        Bayesian Linear Regression model
        
        Parameters:
        -----------
        alpha: float
            Precision (inverse variance) of the prior distribution on weights
        beta: float
            Precision (inverse variance) of the noise
        """
        self.alpha = alpha  # Prior precision
        self.beta = beta    # Noise precision
        self.mean = None    # Posterior mean
        self.precision = None  # Posterior precision
        self.inv_precision = None  # cached inverse precision [d,d]
        
    def fit(self, X, y):
        """
        Fit the Bayesian linear regression model
        
        Parameters:
        -----------
        X: torch.Tensor of shape (n, d)
            Training data
        y: torch.Tensor of shape (n,)
            Target values
        """
        n= X.shape[0]  # int n
        d= X.shape[1]  # int d
        
        # posterior precision: alpha*I_d + beta*(X^T X) -> [d,d]
        self.precision = self.alpha * torch.eye(d, device=device) + self.beta * (X.T @ X)  # [d,d]

        # cache inverse precision once -> [d,d]
        self.inv_precision = torch.linalg.inv(self.precision)  # [d,d]
        # posterior mean: beta * inv_precision * (X^T y) -> [d]
        self.mean = self.beta * (self.inv_precision @ (X.T @ y))  # [d]
        return self
        
    def predict(self, X, return_std=False):
        """
        Make predictions with the model
        Inputs
        - X: torch.Tensor [B, d] on device; test data
        - return_std: bool; whether to return predictive std
        Returns
        - mean: torch.Tensor [B] on device
        - std: torch.Tensor [B] on device (if return_std=True)
        """
        batch_size= X.shape[0]   # int B
        d= X.shape[1]            # int d
        mean = X @ self.mean     # [B,d]@[d] -> [B]
        
        if return_std:
            # var = 1/beta + diag(X inv_precision X^T) -> [B]
            var = (1/self.beta)* torch.ones(batch_size, device=device) + torch.sum((X @ self.inv_precision) * X, dim=1)  # [B]
            std = torch.sqrt(var)  # [B]
            return mean, std
        return mean

def run_simulation(n, d, S, w_T, w_R, sigma, gamma, n_test=1):
    """
    Run the simulation for Bayesian Linear Regression.
    Inputs (CPU unless stated)
    - n: int, number of training samples
    - d: int, number of features
    - S: float, std of input features
    - w_T: torch.Tensor [d] on device, teacher weights
    - w_R: torch.Tensor [d] on device, reward weights
    - sigma: float, noise std
    - gamma: float, prior std
    - n_test: int, number of test samples
    Returns (on device)
    - diff_exp: torch.Tensor [n_test]; experimental difference in prediction: mean of y - y_test
    - diff_th: torch.Tensor [n_test]; theoretical difference: m - y_test
    - s: torch.Tensor [n_test]; experimental predictive std
    - y_TR: torch.Tensor [n_test]; y_T - y_R
    """
    # Generate training data X ~ N(0,S^2)
    X = torch.normal(mean=0, std=S, size=(n,d), device=device)      # [n,d]
    X= X/(d**(1/2))  # [n,d]
    y = X@w_T + sigma * torch.randn(n, device=device)               # [n]

    # Fit Bayesian linear regression
    model = BayesianLinearRegression(alpha=1/gamma**2, beta=1/sigma**2)  # scalars on device via ops
    model.fit(X, y)  # sets mean:[d], precision:[d,d]

    # Make predictions with uncertainty on test data
    generator=torch.Generator(device=device)
    X_test = torch.normal(mean=0, std=S, size=(n_test,d), device=device, generator=generator)/(d**(1/2))  # [n_test,d]
    y_test = X_test @ w_T   # [n_test]
    y_TR = X_test @ w_T - X_test @ w_R  # [n_test]
    m, s = model.predict(X_test, return_std=True)  # m:[n_test], s:[n_test]
    

    # DE predictions 
    Rh=d*sigma**2/(n*(gamma**2))  # scalar
    R=(1/2)*(S**2)*(d/n+Rh/S**2-1+((d/n+Rh/S**2-1)**2+4*Rh/S**2)**(1/2))  # scalar
    diff_exp=(m-y_test)  # [n_test]
    diff_th=-R/(R+S**2)*y_test  # [n_test]
    S_expt = X_test.square().sum(dim=-1).sqrt()  # [n_test]
    s_th = (sigma * sigma + (gamma * gamma) * R * (S_expt * S_expt) / (R + S * S)).sqrt()  # [n_test]

    return diff_exp, diff_th, s, s_th, y_TR  # [n_test], [n_test], [n_test], [n_test]

def check_extreme_value_distribution(sample_size, num_trials, mean, std,  mean_th, std_th, y_TR, trials_per_batch=256, k_chunk_size=128):
    """
    Empirical vs theoretical value of generation error
    Inputs
    - sample_size: int (CPU), k
    - num_trials: int (CPU), total number of trials per sample
    - mean: torch.Tensor [B] on device; experimental mean per test sample
    - std: torch.Tensor [B] on device; experimental std per test sample
    - mean_th: torch.Tensor [B] on device; theoretical mean per test sample
    - std_th: torch.Tensor [B] or scalar on device; theoretical std per test sample
    - y_TR: torch.Tensor [B] on device; y_T - y_R per test sample
    - trials_per_batch: int (CPU); number of trials processed per batch
    - k_chunk_size: int (CPU); number of k processed per chunk to cap memory
    Returns
    - empirical_mean: torch.Tensor [B] on device; empirical delta(k)
    - theoretical_mean: torch.Tensor [B] on device; theoretical delta(k)
    """
    ref_dtype = mean.dtype if isinstance(mean, torch.Tensor) else torch.get_default_dtype()
    def to_dev_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=ref_dtype)
        else:
            return torch.tensor(x, device=device, dtype=ref_dtype)

    mean    = to_dev_tensor(mean)
    std     = to_dev_tensor(std)
    mean_th = to_dev_tensor(mean_th)
    std_th  = to_dev_tensor(std_th)
    y_TR    = to_dev_tensor(y_TR)

    B = mean.shape[0]  # batch size

    # Running weighted sum of empirical means across trial batches
    empirical_sum = torch.zeros(B, device=device, dtype=ref_dtype)
    total_trials = 0

    
    for tr_start in range(0, num_trials, trials_per_batch):
        Tb = min(trials_per_batch, num_trials - tr_start)  # trials in this batch
        # Initialize running best over k-chunks for this trials batch
        best_vals = torch.full((B, Tb), -torch.inf, device=device, dtype=ref_dtype)  # stores max of -(sample+y_TR)^2
        best_sample = torch.zeros(B, Tb, device=device, dtype=ref_dtype)             # stores corresponding sample value

        # Stream over k in chunks
        for k_start in range(0, sample_size, k_chunk_size):
            Kc = min(k_chunk_size, sample_size - k_start)
            # eps ~ N(0,1) -> [B,Tb,Kc]
            eps = torch.randn(B, Tb, Kc, device=device, dtype=ref_dtype)
            # samples = mean + std*eps -> [B,Tb,Kc]
            samples = mean.view(B,1,1) + std.view(B,1,1) * eps
            # candidates for maximization: -(samples + y_TR)^2 -> [B,Tb,Kc]
            candidates = -(samples + y_TR.view(B,1,1)).pow(2)
            # local maxima over this chunk -> [B,Tb]
            local_max_vals, local_idx = candidates.max(dim=-1)
            # gather corresponding sample (y_i) values -> [B,Tb]
            local_best_sample = samples.gather(-1, local_idx.unsqueeze(-1)).squeeze(-1)
            # update global bests
            mask = local_max_vals > best_vals
            best_vals = torch.where(mask, local_max_vals, best_vals)
            best_sample = torch.where(mask, local_best_sample, best_sample)

        # After all k-chunks, compute batch empirical mean over trials -> [B]
        empirical_mean_batch = best_sample.pow(2).mean(dim=-1)
        empirical_sum += empirical_mean_batch * Tb
        total_trials += Tb

    # Average over all trials -> [B]
    empirical_mean = empirical_sum / total_trials

    # Theory c_n and mean -> [B]
    n_k = torch.tensor(sample_size, device=device, dtype=ref_dtype)
    c_n =(torch.pi/2) / (n_k**2) * torch.exp((mean_th**2) / (std_th**2))  # [B]
    theoretical_mean = 2 * (std_th**2) * c_n                              # [B]

    return empirical_mean, theoretical_mean  # [B], [B]

def run_inferences(n, d, S, w_T, w_R, sigma, gamma, n_test, sample_size_start, sample_size_end, num_trials):
    """
    End-to-end pipeline: fit BLR, get per-sample predictive stats, and compute extreme-value means.
    Inputs (CPU unless stated)
    - n: int, number of training samples
    - d: int, number of features
    - S: float, std of input data
    - w_T: torch.Tensor [d] on device, teacher weights
    - w_R: torch.Tensor [d] on device, reward weights
    - sigma: float, noise std
    - gamma: float, prior std
    - n_test: int, number of test samples
    - sample_size_start: int, minimum k
    - sample_size_end: int, maximum k (exclusive)
    - num_trials: int, number of trials per k
    Returns (CPU tensors)
    - empirical_mean_arr: list of torch.Tensor scalars [1] on CPU (per k), averaged over B
    - theoretical_mean_arr: list of torch.Tensor scalars [1] on CPU (per k), averaged over B
    """

    diff_exp, diff_th, s, s_th, y_TR=run_simulation(n, d, S, w_T, w_R, sigma, gamma, n_test)  # each [n_test]

    # Experimental and theoretical parameters per test sample -> [n_test]
    mean = diff_exp                           # [n_test]
    mean_th= diff_th                          # [n_test]
    std =  s                                  # [n_test]
    std_th= s_th                              # [n_test]

    empirical_mean_arr= []    # list of scalar tensors (CPU) per k
    theoretical_mean_arr= []  # list of scalar tensors (CPU) per k
    for i in range(sample_size_start, sample_size_end):
        empirical_mean, theoretical_mean = check_extreme_value_distribution(i, num_trials, mean, std, mean_th, std_th, y_TR)  # each [n_test]
        empirical_mean_arr.append(empirical_mean.mean(dim=-1).to("cpu"))      # scalar CPU
        theoretical_mean_arr.append(theoretical_mean.mean(dim=-1).to("cpu"))  # scalar CPU
    
    return empirical_mean_arr, theoretical_mean_arr

num_trials = 10**2    # Number of trials to find maxima for given k
#n=10**7 # Number of training samples
d=1*10**1 # Number of features
S=1 # Standard deviation of the input data
sigma=10**(-4)    # Noise std
gamma=1*10**(-3)  # Prior std
n_test=10**5 # Number of test samples for BLR, number of x to average over
w_mean=0
w_std=2
w_T = torch.normal(mean=w_mean, std=w_std, size=(d,), device=device) # Teacher weights
sample_size_start = 50      # k min
sample_size_end = 200       # k max (exclusive) 
c=0 # c


plt.figure(figsize=(10, 6))
x_values = list(range(sample_size_start, sample_size_end))
scale_factor=1/sigma**2
n_arr=np.array([1*10**4, 1*10**4+5*10**3, 2*10**4, 2*10**4+5*10**3,3*10**4, 3.5*10**4, 4*10**4])
exp_norm = plt.Normalize(vmin=n_arr.min(), vmax=n_arr.max())
exp_map = plt.cm.Reds_r  
th_norm = plt.Normalize(vmin=n_arr.min(), vmax=n_arr.max())
th_map = plt.cm.Blues_r  
n_list = [int(x) for x in n_arr[:-2]]
for n in n_list:
    print(f"Running for n={n}, d={d}, sigma={sigma}, gamma={gamma}, S={S}, T/(2*sigma^2)={0}, c={c}")
    Rh = d * sigma**2 / (n * (gamma**2))
    R = (1/2) * (S**2) * (d/n + Rh/S**2 - 1 + ((d/n + Rh/S**2 - 1)**2 + 4*Rh/S**2)**(1/2))
    w_R = (1 + c*R) * w_T

    # Experimental (empirical) via simulation
    empirical_mean_arr, _ = run_inferences(n, d, S, w_T, w_R, sigma, gamma, n_test, sample_size_start, sample_size_end, num_trials)

    # Theoretical array per k
    theoretical_mean_arr = []
    for i in range(sample_size_start, sample_size_end):
        theoretical_mean = 2 * (sigma**2) * ((np.pi/2) / i**2) / torch.sqrt(
            1 - ((2 * torch.sum(w_T**2)) / (d * gamma**2)) * ((d/n) * (sigma / (gamma * S)))**2
        )
        theoretical_mean_arr.append(theoretical_mean.to("cpu"))

    # Scale and plot
    empirical_mean_arr = [x * scale_factor for x in empirical_mean_arr]
    theoretical_mean_arr = [x * scale_factor for x in theoretical_mean_arr]

    color_th = th_map(th_norm(n))
    color_exp = exp_map(exp_norm(n))
    plt.plot(x_values, empirical_mean_arr, color=color_exp, alpha=1, label='Expt. at n='+str(n))
    plt.plot(x_values, theoretical_mean_arr, color=color_th, alpha=1, label='L.T.E. at n='+str(n))
    

fontsize=16
plt.xlabel('$k$', fontsize=fontsize)
plt.ylabel('$\\delta/\\sigma^2 $', fontsize=fontsize)
ax = plt.gca()  # Get current axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95), fontsize=fontsize)
plt.grid(True, alpha=0.3)
plt.tick_params(axis='both', which='major', labelsize=fontsize)
plt.savefig('BLR_d'+str(d)+'_t0.png', dpi=300, bbox_inches='tight')
plt.show()
