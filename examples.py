import numpy as np
import algorithm as alg
import matplotlib.pyplot as plt
from scipy import linalg
tensor = np.random.randn(3, 4, 3)
tensor_ranks = (2, 2, 2)
factors = []
for n in range(3):
    factors.append(linalg.orth(np.random.randn(tensor.shape[n], tensor_ranks[n])))
# Example 1:
# ====L1-HOOI with fixed-point underlying L1-PCA solver
core1, factors1, metricEvolution1 = alg.l1hooi(tensor, tensor_ranks, factors, solver = 'fixedpoint')

# Example 2:
# ====L1-HOOI with bitflipping underlying L1-PCA solver
core2, factors2, metricEvolution2 = alg.l1hooi(tensor, tensor_ranks, factors, solver = 'bitflipping')

# Example 3:
# ====L1-HOOI with exactpoly underlying L1-PCA solver
core3, factors3, metricEvolution3 = alg.l1hooi(tensor, tensor_ranks, factors, solver = 'exactpoly')

# Example 4:
# ====L1-HOOI with exact underlying L1-PCA solver
core4, factors4, metricEvolution4 = alg.l1hooi(tensor, tensor_ranks, factors, solver = 'exact')


plt.figure()
plt.plot(metricEvolution1, '--r', label = "Underlying L1-PCA solver: fixed-point")
plt.plot(metricEvolution2, ':b', label = "Underlying L1-PCA solver: bit-flipping")
plt.plot(metricEvolution3, ':k', label = "Underlying L1-PCA solver: exact-poly")
plt.plot(metricEvolution4, '--k', label = "Underlying L1-PCA solver: exact")
plt.ylabel('L1-Tucker metric')
plt.xlabel('Iteration index')
plt.legend()
plt.show()