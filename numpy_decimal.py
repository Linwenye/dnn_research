import numpy as np

from decimal import Decimal, getcontext

# W1 = np.random.randn(2, 4).astype(np.object) * Decimal(0.01)
# W1 = np.random.randn(2, 4) * 0.01
# W_1 = np.array(W1, dtype=np.longdouble)

# print(type(W_1[0][0]))
# print(W1)
# print(W_1)

from mpmath import mpf, mp


def to_mp(original, precision=53):
    return original * mpf('1.0')


def set_mp_precision(precision):
    mp.prec = precision


def to_decimal(original):
    ori_shape = original.shape
    return np.asarray([Decimal(x) for x in original.flat]).reshape(ori_shape)


def set_decimal_precision(precision):
    getcontext().prec = precision


def decimal_log(original):
    ori_shape = original.shape
    return np.asarray([x.ln() for x in original.flat]).reshape(ori_shape)


if __name__ == '__main__':
    # a = [[Decimal('1.3'), Decimal('2.6')], [Decimal('0.3'), Decimal(1.6)]]
    # a_np = np.asarray(a)
    # print(a_np.shape)
    # print(np.exp(a))
    a = np.random.rand(2, 3)
    b = to_decimal(a)
    b = decimal_log(b)
    print(b)
