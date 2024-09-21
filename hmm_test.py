import numpy as np
from hmmlearn import hmm

# Generate sample data
X = np.random.rand(100, 2)

# Initialize the Gaussian HMM
model = hmm.GaussianHMM(n_components=3, n_iter=100)

# Fit the model
model.fit(X)

print("Model fitted successfully.")
