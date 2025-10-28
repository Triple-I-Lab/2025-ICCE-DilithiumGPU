import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import time
from dilithium_gpu import *

# Assuming all the necessary functions and constants are defined above
def run_benchmark(mode, batch_sizes):
    keygen_results = []
    sign_results = []
    verify_results = []

    for batch_size in batch_sizes:
        global BATCH_SIZE, MODE, PARAMS
        BATCH_SIZE = batch_size
        MODE = mode
        PARAMS = DILITHIUM_MODES[MODE]

        # Key generation benchmark
        A, s1, s2, t, key_gen_time = batch_key_gen(BATCH_SIZE)
        keygen_throughput = BATCH_SIZE / key_gen_time
        keygen_results.append(keygen_throughput)

        # Signing benchmark
        messages = [f"Message {i}" for i in range(BATCH_SIZE)]
        z, c, h, valid_mask, sign_time = batch_sign(messages, A, s1, s2)
        sign_throughput = BATCH_SIZE / (sign_time)
        sign_results.append(sign_throughput)

        # Verification benchmark
        verify_results_data, verify_time, z_ok, h_ok, c_ok = batch_verify(messages, (z, c, h), A, t)
        verify_throughput = BATCH_SIZE / verify_time
        verify_results.append(verify_throughput)

    return keygen_results, sign_results, verify_results

def plot_throughput():
    batch_sizes = [1, 10, 50, 100, 200,300,400, 500,600,700,800,900, 1000, 2000]
    modes = [2, 3, 5]
    operations = ['Key Gen', 'Sign', 'Verify']

    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for i, operation in enumerate(operations):
        for mode in modes:
            if operation == 'Key Gen':
                data = run_benchmark(mode, batch_sizes)[0]
            elif operation == 'Sign':
                data = run_benchmark(mode, batch_sizes)[1]
            else:
                data = run_benchmark(mode, batch_sizes)[2]

            axs[i].plot(batch_sizes, data, label=f'Dilithium-{mode}')

        axs[i].set_xscale('log')
        axs[i].set_yscale('log')
        axs[i].set_xlabel('Batch Size')
        axs[i].set_title(operation)
        axs[i].legend()
        axs[i].grid(True)

    print(f'{data=}, {mode=}, {operation}')
    axs[0].set_ylabel('Throughput (operations/second)')

    plt.suptitle('Dilithium Throughput for Key Gen, Sign and Verify')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('dilithium_throughput.png')
    plt.show()


if __name__ == "__main__":
    plot_throughput()

def calculate_averages():
    batch_sizes = [200]
    modes = [2, 3, 5]
    operations = ['Gen', 'Sign', 'Verify']

    averages = {mode: {op: [] for op in operations} for mode in modes}

    for _ in range(1):  # Run 100 iterations
        for mode in modes:
            keygen_results, sign_results, verify_results = run_benchmark(mode, batch_sizes)
            averages[mode]['Gen'].append(keygen_results)
            averages[mode]['Sign'].append(sign_results)
            averages[mode]['Verify'].append(verify_results)

    # Calculate the average throughput for each mode and operation
    # final_averages = {mode: {op: np.mean(averages[mode][op]) for op in operations} for mode in modes}

    return averages

def run_average():
    # Get the averages
    averages = calculate_averages()

    # Print results
    for mode in averages:
        print(f"Mode {mode}:")
        for procedure in averages[mode]:
            print(f"  {procedure}: {averages[mode][procedure]:.2f} op/s")

# run_average()

def run_benchmark(mode, batch_size):
    global MODE, PARAMS, BATCH_SIZE
    MODE = mode
    PARAMS = DILITHIUM_MODES[MODE]
    BATCH_SIZE = batch_size

    start_time = time.time()
    A, s1, s2, t, _ = batch_key_gen(BATCH_SIZE)
    messages = [f"Message {i}" for i in range(BATCH_SIZE)]
    z, c, h, _, _ = batch_sign(messages, A, s1, s2)
    _, _, _, _, _ = batch_verify(messages, (z, c, h), A, t)
    end_time = time.time()

    total_time = end_time - start_time
    latency = total_time / BATCH_SIZE  # Average latency per operation
    return latency


def plot_latency():
    batch_sizes = [1, 10, 100, 500, 1000, 5000, 10000]
    modes = [2, 3, 5]

    data = []
    for mode in modes:
        mode_data = []
        for batch_size in batch_sizes:
            latency = run_benchmark(mode, batch_size)
            mode_data.append(latency)
        data.append(mode_data)

    x = np.arange(len(batch_sizes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    rects1 = ax.bar(x - width, data[0], width, label='Dilithium-2')
    rects2 = ax.bar(x, data[1], width, label='Dilithium-3')
    rects3 = ax.bar(x + width, data[2], width, label='Dilithium-5')

    ax.set_ylabel('Latency (seconds)')
    ax.set_xlabel('Batch Size')
    ax.set_title('Dilithium Latency Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=90)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()
    plt.savefig('group_latency.png')
    plt.show()


# if __name__ == "__main__":
#     plot_latency()

import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import time
import psutil
import os


# Assuming all the necessary functions and constants are defined above

def measure_memory_usage(mode, batch_size):
    global MODE, PARAMS, BATCH_SIZE
    MODE = mode
    PARAMS = DILITHIUM_MODES[MODE]
    BATCH_SIZE = batch_size

    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss / 1024 / 1024  # Memory usage in MB

    A, s1, s2, t, _ = batch_key_gen(BATCH_SIZE)
    messages = [f"Message {i}" for i in range(BATCH_SIZE)]
    z, c, h, _, _ = batch_sign(messages, A, s1, s2)
    _, _, _, _, _ = batch_verify(messages, (z, c, h), A, t)

    end_memory = process.memory_info().rss / 1024 / 1024  # Memory usage in MB
    memory_usage = end_memory - start_memory

    return memory_usage


def plot_memory_usage():
    batch_sizes = [1, 10, 100, 500, 1000, 5000, 10000]
    modes = [2, 3, 5]

    data = []
    for mode in modes:
        mode_data = []
        for batch_size in batch_sizes:
            memory_usage = measure_memory_usage(mode, batch_size)
            mode_data.append(memory_usage)
        data.append(mode_data)

    x = np.arange(len(batch_sizes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    rects1 = ax.bar(x - width, data[0], width, label='Dilithium-2')
    rects2 = ax.bar(x, data[1], width, label='Dilithium-3')
    rects3 = ax.bar(x + width, data[2], width, label='Dilithium-5')

    ax.set_ylabel('Memory Usage (MB)')
    ax.set_xlabel('Batch Size')
    ax.set_title('Dilithium Memory Usage Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=90)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()
    plt.savefig('group_memory.png')
    plt.show()


# if __name__ == "__main__":
#     plot_memory_usage()


import time
import numpy as np


def analyze_performance(num_trials=100):
    results = {
        'Key Generation': [],
        'Signature': [],
        'Verification': []
    }

    for _ in range(num_trials):
        print(_)
        # Key Generation
        start_time = time.time()
        A, s1, s2, t, _ = batch_key_gen(BATCH_SIZE)
        key_gen_time = time.time() - start_time
        results['Key Generation'].append(key_gen_time)

        # Signature
        messages = [f"Message {i}" for i in range(BATCH_SIZE)]
        start_time = time.time()
        z, c, h, valid_mask, _ = batch_sign(messages, A, s1, s2)
        sign_time = time.time() - start_time
        results['Signature'].append(sign_time)

        # Verification
        start_time = time.time()
        _, _, _, _, _ = batch_verify(messages, (z, c, h), A, t)
        verify_time = time.time() - start_time
        results['Verification'].append(verify_time)

    for operation, times in results.items():
        min_time = np.min(times) * 1000 / BATCH_SIZE # Convert to milliseconds
        avg_time = np.mean(times) * 1000 / BATCH_SIZE
        max_time = np.max(times) * 1000 / BATCH_SIZE
        print(f"{operation}:")
        print(f"  Min: {min_time:.3f} ms")
        print(f"  Ave: {avg_time:.3f} ms")
        print(f"  Max: {max_time:.3f} ms")


# if __name__ == "__main__":
#     analyze_performance()