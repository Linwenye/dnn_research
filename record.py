import time
from matplotlib import pyplot as plt

time_records = []
start = time.time()


def record_time():
    time_elapse = time.time() - start
    print(time_elapse)
    time_records.append(time_elapse)


def plot_time():
    plt.plot(time_records)
    plt.ylabel('time(s)')
    plt.xlabel('iterations(per 300)')
    plt.show()


def print_record():
    print(time_records)
