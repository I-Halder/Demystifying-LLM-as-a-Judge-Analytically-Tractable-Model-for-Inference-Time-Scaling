"""
Bayesian Linear Regression (BLR) with quadratic reward function.

Notation:
- n: int, number of training samples (CPU int)
- d: int, number of features (CPU int)
- Bt: int, current batch size of test samples (CPU int per iteration)
- Tb: int, current batch size of trials (CPU int per iteration)
- Kmax: int, maximum k value (CPU int)
- ks: torch.Tensor [K_sel] on device, range of k values
- K_sel: int, number of elements of ks

"""

import torch
import matplotlib.pyplot as plt
import numpy as np
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BLR trainer
class BayesianLinearRegression:
    def __init__(self, alpha, beta):
        """Initialize BLR model.
        Parameters
        - alpha: torch.Tensor scalar [] on device; prior precision (1/gamma^2)
        - beta: torch.Tensor scalar [] on device; noise precision (1/sigma^2)
        Attributes (on device)
        - mean: torch.Tensor [d], posterior mean of weights
        - precision: torch.Tensor [d,d], posterior precision
        - inv_precision: torch.Tensor [d,d], cached inverse of precision
        """
        self.alpha = alpha  # Prior precision, on device
        self.beta = beta    # Noise precision, on device
        self.mean = None    # Posterior mean, on device
        self.precision = None  # Posterior precision, on device
        self.inv_precision = None  # Inverse precision to avoid recomputation, on device

    def fit(self, X, y):
        """Fit BLR posterior.
        Inputs
        - X: torch.Tensor [n, d] on device; training features
        - y: torch.Tensor [n] on device; training targets
        Returns
        - self: fitted model with
            mean: [d] on device
            precision: [d,d] on device
            inv_precision: [d,d] on device
        """
    
        n= X.shape[0]  # int
        d= X.shape[1]  # int
        
        # Calculate posterior precision: alpha*I_d + beta*X^T X -> [d,d]
        self.precision = self.alpha * torch.eye(d, device=device) + self.beta * (X.T @ X)  # [d,d]

        # Posterior mean: beta * precision^{-1} X^T y -> [d]
        self.mean = self.beta * torch.linalg.inv(self.precision) @ (X.T @ y)  # [d]
        # Inverse for predictive variance -> [d,d]
        self.inv_precision = torch.linalg.inv(self.precision)  # [d,d]
        return self
        
    def predict(self, X, return_std=False):
        """Predict posterior predictive mean/std.
        Inputs
        - X: torch.Tensor [B, d] on device; test features
        - return_std: bool; whether to return predictive std
        Returns
        - mean: torch.Tensor [B] on device
        - std: torch.Tensor [B] on device (if return_std=True)
        """
        batch_size= X.shape[0]  # int B
        d= X.shape[1]           # int d
        mean = (X @ self.mean)  # [B,d]@[d] -> [B]
        
        if return_std:
            # var = 1/beta + diag(X inv_precision X^T) -> [B]
            var = (1/self.beta)* torch.ones(batch_size, device=device) \
                  + torch.sum((X @ self.inv_precision) * X, dim=1)  # [B,d]*[B,d] sum-> [B]
            std = torch.sqrt(var)  # [B]
            return mean, std
        return mean

# Small test batch generator
@torch.no_grad()
def generate_test_batches(model, n_test, d, S, w_T, sigma, gamma, R, batch_size, reward_pram=None):
    """Stream test data and predictive stats on device.
    Inputs
    - model: BayesianLinearRegression on device, trained
    - n_test: int (CPU), total number of test samples to generate
    - d: int (CPU), number of features
    - S: float (CPU), feature std; 
    - w_T: torch.Tensor [d] on device, teacher weights
    - sigma: float (CPU), noise std; 
    - gamma: float (CPU), prior std; 
    - R: float (CPU), renormalized ridge
    - batch_size: int (CPU), number of test samples per generated batch
    - reward_pram: Optional[float] (CPU), c 
    Yields (each on device, shape [Bt])
    - mean_expt: torch.Tensor [Bt], m_expt
    - std_expt: torch.Tensor [Bt], s_expt
    - mean_th: torch.Tensor [Bt], m as given by DE
    - std_th: torch.Tensor [Bt], s as given by DE
    - mR: torch.Tensor [Bt], \mu_R
    - mT: torch.Tensor [Bt], \mu_T
    """
    device = w_T.device  
    dtype =  torch.float32

    
    S_t = torch.as_tensor(S, device=device, dtype=dtype)          # []
    sigma_t = torch.as_tensor(sigma, device=device, dtype=dtype)  # []
    gamma_t = torch.as_tensor(gamma, device=device, dtype=dtype)  # []
    R_t = torch.as_tensor(R, device=device, dtype=dtype)          # []
    d_t = torch.as_tensor(d, device=device, dtype=dtype)          # []
    scale_d = d_t.sqrt()                                          # []
    reward_t = None if reward_pram is None else torch.as_tensor(reward_pram, device=device, dtype=dtype)  # [] or None

    gen = torch.Generator(device=device)

    for start in range(0, n_test, batch_size):  
        Bt = min(batch_size, n_test - start)  # int
        X_test = torch.normal(0.0, S_t, size=(Bt, d), device=device, generator=gen) / scale_d  # [Bt,d]
        m, s = model.predict(X_test, return_std=True)  # m:[Bt], s:[Bt]
        mT = X_test @ w_T  # [Bt,d]@[d] -> [Bt]
        mR = mT            # [Bt]
        if reward_t is not None:
            mR = mR - reward_t * (m - mT)  # [Bt]
        S_expt = X_test.square().sum(dim=-1).sqrt()  # [Bt]
        mean_expt = m                                # [Bt]
        mean_th = (1 - R_t / (R_t + S_t * S_t)) * mT # [Bt]
        std_expt = s                                 # [Bt]
        std_th = (sigma_t * sigma_t + (gamma_t * gamma_t) * R_t * (S_expt * S_expt) / (R_t + S_t * S_t)).sqrt()  # [Bt]
        yield mean_expt, std_expt, mean_th, std_th, mR, mT  # each [Bt]

@torch.no_grad()
def calculate_generalization_error(batch_iter, temp, ks, num_trials, trials_per_batch=128):
    """Compute empirical & theoretical generalization error.
    Inputs
    - batch_iter: iterator yielding tuples (mean_expt, std_expt, mean_th, std_th, mR_b, mT_b), each [Bt] on device
    - temp: torch.Tensor scalar [] on device; T
    - ks: torch.Tensor [K_sel] on device; all the values of k we want to look at
    - num_trials: int (CPU); total number of Monte Carlo trials per test sample
    - trials_per_batch: int (CPU); number of trials to process per inner batch
    Returns (CPU tensors)
    - empirical_val: torch.Tensor [K_sel] on CPU; empirical delta
    - LTE_val: torch.Tensor [K_sel] on CPU; Series expansion of delta
    - DE_val: torch.Tensor [K_sel] on CPU; delta(k) DE version of delta
    """
    device = temp.device                 # device
    dtype = temp.dtype                   # dtype for created tensors
    Kmax = int(ks.max().item())          # int
    ks = ks.to(device=device)            # [K_sel]
    ks_long = ks.to(torch.long)          # [K_sel]
    ks_list = [int(k) for k in ks_long.tolist()]  # list length K_sel
    pos = {k: i for i, k in enumerate(ks_list)}   # tensor as key is not allowed
    K_sel = ks_long.shape[0]                                 # int
    empirical_sum = torch.zeros(K_sel, device=device, dtype=dtype)  # [K_sel]
    DE_sum = torch.zeros(K_sel, device=device, dtype=dtype)         # [K_sel]
    total_test_samples = 0                                          # int

    # Streaming accumulators for theory means A0, B0, C0 (all scalars on device)
    A0_sum = torch.zeros((), device=device, dtype=dtype)  # []
    B0_sum = torch.zeros((), device=device, dtype=dtype)  # []
    C0_sum = torch.zeros((), device=device, dtype=dtype)  # []
    D0_sum = torch.zeros((), device=device, dtype=dtype)  # []

    two = torch.as_tensor(2.0, device=device, dtype=dtype)  # []
    three = torch.as_tensor(3.0, device=device, dtype=dtype)  # []

    for mean_expt, std_expt, mean_th, std_th, mR_b, mT_b in batch_iter:  # each [Bt]
        # Ensure batch tensors are on device/dtype
        mean_expt = mean_expt.to(device=device, dtype=dtype)  # [Bt]
        std_expt = std_expt.to(device=device, dtype=dtype)    # [Bt]
        mean_th = mean_th.to(device=device, dtype=dtype)      # [Bt]
        std_th = std_th.to(device=device, dtype=dtype)        # [Bt]
        mR_b = mR_b.to(device=device, dtype=dtype)            # [Bt]
        mT_b = mT_b.to(device=device, dtype=dtype)            # [Bt]

        Bt = mean_expt.shape[0]  # int

        # Low temperature expansion -> [Bt]
        A0_b = (mean_th - mT_b).pow(2) + std_th.pow(2)  # [Bt]
        B0_b = -(two * std_th.pow(2) / temp) * (two * (mean_th - mR_b) * (mean_th - mT_b) + std_th.pow(2))  # [Bt]
        C0_b = (two * std_th.pow(2) / temp).pow(2) * ((mean_th - mR_b) * (two * (mean_th - mT_b) + (mean_th - mR_b)) + std_th.pow(2))  # [Bt]
        D0_b = -(two * std_th.pow(2) / temp).pow(3) * ((mean_th - mR_b) * (two * (mean_th - mT_b) + 2*(mean_th - mR_b)) + std_th.pow(2))  # [Bt]
        A0_sum += A0_b.sum()  # []
        B0_sum += B0_b.sum()  # []
        C0_sum += C0_b.sum()  # []
        D0_sum += D0_b.sum()  # []

        # Streaming DE/empirical means
        def compute_means_for_pair(mean_b, std_b, mR_b, mT_b):
            """Compute per-k errors averaged over trials for one batch.
            Inputs: mean_b,std_b,mR_b,mT_b each [Bt] on device
            Returns: torch.Tensor [Bt, K_sel] on device
            """
            mean_b = mean_b.to(device=device, dtype=dtype)  # [Bt]
            std_b = std_b.to(device=device, dtype=dtype)    # [Bt]
            mR_b = mR_b.to(device=device, dtype=dtype)      # [Bt]
            mT_b = mT_b.to(device=device, dtype=dtype)      # [Bt]

            errs_means_weighted_sum = torch.zeros(Bt, K_sel, device=device, dtype=dtype)  # [Bt,K_sel]
            total_trials_b = 0  # int
            for tr_start in range(0, num_trials, trials_per_batch):  # CPU loop
                Tb = min(trials_per_batch, num_trials - tr_start)  # int
                cum_w = torch.zeros(Bt, Tb, device=device, dtype=dtype)    # [Bt,Tb]
                cum_num = torch.zeros(Bt, Tb, device=device, dtype=dtype)  # [Bt,Tb]
                # Store per-k mean over trials for this trial batch -> [Bt,K_sel]
                errs_k_batch = torch.zeros(Bt, K_sel, device=device, dtype=dtype)  # [Bt,K_sel]
                for k in range(1, Kmax + 1):  # sequential over k (CPU loop)
                    eps = torch.randn(Bt, Tb, device=device, dtype=dtype)  # [Bt,Tb]
                    samples = mean_b.view(Bt, 1) + std_b.view(Bt, 1) * eps  # [Bt,1]+[Bt,1]*[Bt,Tb]->[Bt,Tb]
                    weight = torch.exp(-(samples - mR_b.view(Bt, 1)).pow(2) / temp)  # [Bt,Tb]
                    diff_sq = (samples - mT_b.view(Bt, 1)).pow(2)  # [Bt,Tb]
                    cum_w += weight       # [Bt,Tb]
                    cum_num += diff_sq * weight  # [Bt,Tb]
                    idx = pos.get(k, None)  # CPU int or None
                    if idx is not None:
                        errs = (cum_num / cum_w).mean(dim=1)  # [Bt]
                        errs_k_batch[:, idx] = errs           # [Bt,K_sel]
                errs_means_weighted_sum += errs_k_batch * Tb  # [Bt,K_sel]
                total_trials_b += Tb                           # int
            return errs_means_weighted_sum / total_trials_b     # [Bt,K_sel]

        # Empirical branch -> [Bt,K_sel] -> sum over Bt -> [K_sel]
        emp_running = compute_means_for_pair(mean_expt, std_expt, mR_b, mT_b)  # [Bt,K_sel]
        empirical_sum += emp_running.sum(dim=0)  # [K_sel]
        
        # DE branch -> [Bt,K_sel] -> sum over Bt -> [K_sel]
        de_running = compute_means_for_pair(mean_th, std_th, mR_b, mT_b)  # [Bt,K_sel]
        DE_sum += de_running.sum(dim=0)  # [K_sel]

        total_test_samples += Bt  # int

    # Empirical means across test samples (device)
    empirical_mean_per_k = empirical_sum / total_test_samples  # [K_sel]
    DE_mean_per_k = DE_sum / total_test_samples                # [K_sel]

    # Theoretical averaged across test samples (device)
    A0_mean = A0_sum / total_test_samples  # []
    B0_mean = B0_sum / total_test_samples  # []
    C0_mean = C0_sum / total_test_samples  # []
    D0_mean = D0_sum / total_test_samples  # []
    ks_float = ks.to(device=device, dtype=dtype)  # [K_sel]
    one = torch.ones_like(ks_float)               # [K_sel]
    term1 = (one - one / ks_float)                # [K_sel]
    term2 = term1 * (one - two / ks_float)        # [K_sel]
    term3= term2 * (one - three / ks_float)       # [K_sel]
    LTE_val = A0_mean + B0_mean * term1 + C0_mean * term2 + D0_mean * term3   # [K_sel]
    DE_val = DE_mean_per_k                        # [K_sel]
    empirical_val = empirical_mean_per_k          # [K_sel]
    return empirical_val.detach().cpu(), LTE_val.detach().cpu(), DE_val.detach().cpu()


def run_inferences(n, d, S, w_T, sigma, gamma, n_test, sample_size_start, sample_size_end, num_trials, temp, c=0,
                   trials_per_batch=128, test_batch_size=1000):
    """Train BLR, stream test stats, and compute deltas.
    Inputs
    - n: int (CPU), number of training samples
    - d: int (CPU), number of features
    - S: float (CPU), feature std (training & test)
    - w_T: torch.Tensor [d] on device, teacher weights
    - sigma: float (CPU), noise std
    - gamma: float (CPU), prior std
    - n_test: int (CPU), total test samples to average over
    - sample_size_start: int (CPU), min k (inclusive, 1-based)
    - sample_size_end: int (CPU), max k (exclusive)
    - num_trials: int (CPU), number of trials per test sample
    - temp: float or torch scalar; converted to torch scalar [] on device
    - c: float (CPU), reward parameter
    - trials_per_batch: int (CPU), trials per inner batch
    - test_batch_size: int (CPU), test samples per batch
    Returns (CPU tensors)
    - emp: torch.Tensor [K_sel], empirical delta
    - LTE: torch.Tensor [K_sel], low temperature expansion of delta
    - DE: torch.Tensor [K_sel], delta through deterministic equivalent
    """
    generator = torch.Generator(device=device)
    dtype = torch.float32
    scale_d = torch.sqrt(torch.tensor(d, device=device, dtype=dtype))  # []
    # Training features X ~ N(0,S^2/d) -> [n,d]
    X = torch.normal(0.0, torch.as_tensor(S, device=device, dtype=dtype), size=(n, d), device=device, generator=generator) / scale_d  # [n,d]
    sigma_t = torch.tensor(sigma, device=device, dtype=dtype)  # []
    gamma_t = torch.tensor(gamma, device=device, dtype=dtype)  # []
    # Targets y = X w_T + noise -> [n]
    y = X @ w_T + sigma_t * torch.randn(n, device=device, generator=generator)  # [n]
    # Precisions -> []
    alpha_t = 1.0 / (gamma_t ** 2)  # []
    beta_t = 1.0 / (sigma_t ** 2)   # []
    model = BayesianLinearRegression(alpha=alpha_t, beta=beta_t)
    model.fit(X, y)  # updates mean:[d], precision:[d,d], inv_precision:[d,d]

    # Build a small batch iterator over test data
    Rh = d * sigma**2 / (n * (gamma**2))  # float CPU
    R = 0.5 * (S**2) * (d/n + Rh/S**2 - 1 + ((d/n + Rh/S**2 - 1)**2 + 4 * Rh/S**2)**0.5)  # float CPU
    def batch_iter():
        return generate_test_batches(model, n_test, d, S, w_T, sigma, gamma, R, test_batch_size, reward_pram=c)

    # Run generalization error calculation
    temp=torch.tensor(temp, device=device, dtype=dtype)  # [] on device
    ks = torch.arange(sample_size_start, sample_size_end, device=device, dtype=dtype)  
    emp, LTE, DE = calculate_generalization_error(batch_iter(), temp, ks, num_trials,
                                             trials_per_batch=trials_per_batch)
    return emp, LTE, DE

# Parameters
num_trials = 10**2 # number of trials to calculate \delta for given k, x
n=1*10**4 # number of training data points
d = 10**1 # dim of the data
S = 1 # std of x
sigma = 10**(-4) # noise std
gamma = 1 * 10**(-3) # weight prior std
n_test = 10**6 # number of x to average over for given k and per trial
temp_t = 10 #  T/(2*sigma^2)
temp = temp_t * 2 * sigma**2 # temperature T
w_mean = 0 # mean of teacher weights
w_std = 2   # std of teacher weights
w_T = torch.normal(mean=w_mean, std=w_std, size=(d,), device=device) # teacher weights
sample_size_start = 1 # min k
sample_size_end = 100 # max k

# Plotting
plt.figure(figsize=(10, 6))
x_values = list(range(sample_size_start, sample_size_end))
scaling_factor = 1/sigma**2
c_arr=temp_t*np.arange(1/2,3,1/2)
c_norm = plt.Normalize(vmin=c_arr.min(), vmax=c_arr.max())
cmap = plt.cm.Reds_r 

c_arr=c_arr[:-2] # to maintain visibility in the plot
for c in c_arr:
    print(f"Running for n={n}, d={d}, sigma={sigma}, gamma={gamma}, S={S}, T/(2*sigma^2)={temp_t}, c={c}")
    color = cmap(c_norm(c))
    empirical_mean_arr, theoretical_mean_arr, DE_mean_arr = run_inferences(n, d, S, w_T, sigma, gamma, n_test,sample_size_start, sample_size_end, num_trials, temp, c)
    y_values=[scaling_factor * x for x in empirical_mean_arr]
    plt.plot(x_values, y_values, color=color, alpha=1, label='Expt. at c='+str(c))
    y_DE = [scaling_factor * x for x in DE_mean_arr]
    plt.plot(x_values, y_DE, color=color, linestyle='--', alpha=1, label='D.E. at c='+str(c))
    y_th = [scaling_factor * x for x in theoretical_mean_arr]
    plt.plot(x_values, y_th, color=color, linestyle=':', alpha=1, label='H.T.E. at c='+str(c))

fontsize=20
plt.xlabel('$k$', fontsize=fontsize)
plt.ylabel('$\delta/\sigma^2$', fontsize=fontsize)
ax = plt.gca()  
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))
plt.grid(True, alpha=0.3)
plt.savefig('BLR_n'+str(n)+'_d'+str(d)+'_t'+str(temp_t)+'.png', dpi=300, bbox_inches='tight')
plt.show()


