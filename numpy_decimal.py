import numpy as np

from decimal import Decimal, getcontext

# W1 = np.random.randn(2, 4).astype(np.object) * Decimal(0.01)
# W1 = np.random.randn(2, 4) * 0.01
# W_1 = np.array(W1, dtype=np.longdouble)

# print(type(W_1[0][0]))
# print(W1)
# print(W_1)

from mpmath import mpf, mp

if __name__ == '__main__':
    print(mp)
    mp.prec = 53
    print(mpf(0.3) - mpf(0.1))
    mp.prec = 60
    print(mpf(0.3) - mpf(0.1))
    mp.prec = 100
    print(mpf(0.3) - mpf(0.1))
    a = [mpf('0.3'), mpf('0.1')]
    a_np = np.asarray(a)
    print(a_np)
    print(type(a_np))


def to_mp(original, precision=53):
    return original * mpf('1.0')


def set_mp_precision(precision):
    mp.prec = precision


def to_decimal(original):
    ori_shape = original.shape
    new_array = np.asarray([Decimal(a) for a in original.flat])
    return new_array.reshape(ori_shape)


def set_decimal_precision(precision):
    getcontext().prec = precision
