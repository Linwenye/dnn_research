import time
from matplotlib import pyplot as plt

time_records = []
start = time.time()


def record_time():
    time_records.append(time.time() - start)


def plot_time():
    plt.plot(time_records)
    plt.ylabel('time')
    plt.xlabel('iterations')
    plt.show()


def print_record():
    print(time_records)
