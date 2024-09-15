# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 19:35:24 2024

@author: Saeed
"""

import subprocess
import time

# List of Python scripts to run
scripts = [
    'PL00_scm.py',
    'PL01_inv.py',
    'PL02_unw.py'
]

for script in scripts:
    print(f'Running {script}...')
    
    start_time = time.time()  # Start the timer
    
    process = subprocess.Popen(['python', script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    for line in process.stdout:
        print(line, end='')
    
    process.stdout.close()
    process.wait()
    
    end_time = time.time()  # End the timer
    
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f'{script} finished. Time taken: {elapsed_time:.2f} seconds.\n')