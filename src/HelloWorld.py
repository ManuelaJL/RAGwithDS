import os
import sys

print(f"Python version: {sys.version}")
print(f"Virtual environment: {os.getenv('VIRTUAL_ENV') or 'Not in a virtual environment'}")