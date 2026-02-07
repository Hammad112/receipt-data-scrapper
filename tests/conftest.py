import sys
import os

# Ensure the project root is in the python path for all tests
# This solves the 'ModuleNotFoundError' issues encountered when running tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
