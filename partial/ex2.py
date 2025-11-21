from hmmlearn import hmm
import numpy as np

# definim starile si observatiile
states = ['w','r','s'] #walking running resting
observations = ['l','m','h']

# parametrii
startprob = np.array([0.4, 0.3, 0.3])

transmat = np.array([
    [0.6, 0.3, 0.1],  # w
    [0.2, 0.7, 0.1],  # r
    [0.3, 0.2, 0.5]   # s
])

emissionprob = np.array([
    [0.1, 0.7, 0.2],  # 
    [0.05, 0.25, 0.7], # R emits L, M, H
    [0.8, 0.15, 0.05]  # S emits L, M, H
])

# a)
model = hmm.CategoricalHMM(n_components=3)
model.startprob_ = startprob
model.transmat_ = transmat
model.emissionprob_ = emissionprob
print("A) modelul este gata")

# b)
obs_sequence = np.array([[1, 2, 0]]).T #m ,h ,l
log_prob = model.score(obs_sequence)
prob = np.exp(log_prob)
print(f"b) p(medium,high,low) = {prob}")

# c)
most_likely_states = model.predict(obs_sequence)
print(f"C) cea mai intalnitia secventa: {most_likely_states}")

# d)
n_samples = 10000
sequences = []
for _ in range(n_samples):
    sample, states = model.sample(3)
    if np.array_equal(sample.flatten(), [1, 2, 0]):  # M, H, L
        sequences.append(sample)

e_prob = len(sequences) / n_samples
print(f"empirical probability {e_prob}")
print(f"forward algorithm probability: {prob}")
print(f"difference: {abs(e_prob - prob)}")