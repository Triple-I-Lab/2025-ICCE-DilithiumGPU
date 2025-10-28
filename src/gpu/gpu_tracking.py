import GPUtil
import time

while True:
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU ID: {gpu.id}, GPU Load: {gpu.load*100}%, GPU Memory Free: {gpu.memoryFree}MB, GPU Memory Used: {gpu.memoryUsed}MB")
    time.sleep(10)  # Sleep for 1 second