import numpy as np

from decimal import Decimal

# W1 = np.random.randn(2, 4).astype(np.object) * Decimal(0.01)
W1 = np.random.randn(2, 4) * 0.01
W_1 = np.array(W1, dtype='object')

print(type(W_1[0][0]))
print(W1)
print(W_1)