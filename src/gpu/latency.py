import matplotlib.pyplot as plt
import numpy as np

# Sample data for plotting
num_tasks = np.logspace(1, 6, num=10)
latency_sign_64 = np.random.uniform(10, 100, num_tasks.shape) * num_tasks ** 0.5
latency_verify_64 = np.random.uniform(10, 100, num_tasks.shape) * num_tasks ** 0.5
latency_sign_128 = np.random.uniform(10, 100, num_tasks.shape) * num_tasks ** 0.6
latency_verify_128 = np.random.uniform(10, 100, num_tasks.shape) * num_tasks ** 0.6
latency_sign_256 = np.random.uniform(10, 100, num_tasks.shape) * num_tasks ** 0.7
latency_verify_256 = np.random.uniform(10, 100, num_tasks.shape) * num_tasks ** 0.7

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(num_tasks, latency_sign_64, 'r-', label='Sign(64 threads/block)')
plt.plot(num_tasks, latency_verify_64, 'b-', label='Verify(64 threads/block)')
plt.plot(num_tasks, latency_sign_128, 'g-', label='Sign(128 threads/block)')
plt.plot(num_tasks, latency_verify_128, 'y-', label='Verify(128 threads/block)')
plt.plot(num_tasks, latency_sign_256, 'c-', label='Sign(256 threads/block)')
plt.plot(num_tasks, latency_verify_256, 'm-', label='Verify(256 threads/block)')

# Logarithmic scale
plt.xscale('log')
plt.yscale('log')

# Labels and legend
plt.xlabel('Number of Tasks')
plt.ylabel('Latency (msec)')
plt.title('Latency of Baseline Implementation of Dilithium2 on Testbed-1')
plt.legend()
plt.grid(True)

# Show plot
plt.show()