import csv
import numpy as np
import pymc
from math import log

radon_csv = csv.reader(open('srrs.csv'))
radon = []
for row in radon_csv:
    radon.append(tuple(row))

counties = np.array([x[0] for x in radon])
y = np.array([float(x[1]) for x in radon])
x = np.array([float(x[2]) for x in radon])

## gelman adjustment for log
y[y==0]=.1
y = np.log(y)

## groupings
def createCountyIndex(counties):
    counties_uniq = sorted(set(counties))
    counties_dict = dict()
    for i, v in enumerate(counties_uniq):
        counties_dict[v] = i
    ans = np.empty(len(counties),dtype='int')
    for i in range(0,len(counties)):
        ans[i] = counties_dict[counties[i]]
    return ans

index_c = createCountyIndex(counties)

# Original random effect on intercept
# a = pymc.Normal('a', mu=mu_a, tau=tau_a, value=np.zeros(len(set(counties))))

# Size of truncated DP
N_dp = 50

# Hyperpriors
mu_0 = pymc.Normal('mu_0', mu=0, tau=0.01, value=0)
sig_0 = pymc.Uniform('sig_0', lower=0, upper=100, value=1)
tau_0 = sig_0 ** -2

# Concentration parameter
alpha = pymc.Uniform('alpha', lower=0.3, upper=10)

# Baseline distribution for DP
theta = pymc.Normal('theta', mu=mu_0, tau=tau_0, size=N_dp)

v = pymc.Beta('v', alpha=1, beta=alpha, size=N_dp)
@pymc.deterministic
def p(v=v):
    """ Calculate Dirichlet probabilities """
    
    # Probabilities from betas
    value = [u*np.prod(1-v[:i]) for i,u in enumerate(v)]
    # Enforce sum to unity constraint
    value[-1] = 1-sum(value[:-1])
    
    return value
    
# Expected value of random effect
E_dp = pymc.Lambda('E_dp', lambda p=p, theta=theta: np.inner(p, theta))

z = pymc.Categorical('z', p, size=len(set(counties)))

# Index random effect
a = pymc.Lambda('a', lambda z=z, theta=theta: theta[z])

b = pymc.Normal('b', mu=0., tau=0.0001)

sigma_y = pymc.Uniform('sigma_y', lower=0, upper=100)
tau_y = pymc.Lambda('tau_y', lambda s=sigma_y: s**-2)

# Model
@pymc.deterministic(plot=False)
def y_hat(a=a,b=b):
       return a[index_c] + b*x

# Likelihood
@pymc.stochastic(observed=True)
def y_i(value=y, mu=y_hat, tau=tau_y):
    return pymc.normal_like(value,mu,tau)
