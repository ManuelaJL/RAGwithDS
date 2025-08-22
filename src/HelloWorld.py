import os
import sys

print(f"Python version: {sys.version}")
print(f"Virtual environment: {os.getenv('VIRTUAL_ENV') or 'Not in a virtual environment'}")

from tqdm import tqdm
import time

# for i in tqdm(range(100)):  #tqdm adds a progress bar.
#     time.sleep(0.1)  # Simulate work



import os
print("Basepath: ", ".")
for filename in os.listdir("."):
    full_path = os.path.join(".", filename)
    if os.path.isdir(full_path):
        print("  └─", filename)
    else:
        print(filename)