# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 19:35:24 2024

@author: Saeed
"""

import subprocess
import time

# List of Python scripts to run
scripts = [
    'bin/PL00_scm.py',
    'bin/PL01_inv.py',
    'bin/PL02_unw.py'
]


for script in scripts:
    print(f'Running {script}...')
    
    start_time = time.time()  # Start the timer
    
    # Run the script and capture both stdout and stderr
    process = subprocess.Popen(['python', script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Stream the output in real-time
    for stdout_line in process.stdout:
        print(stdout_line, end='')  # Print stdout as it appears (no extra newlines)

    # Capture and print errors if they occur
    stderr_output = process.stderr.read()
    if stderr_output:
        print(f"Error in {script}:\n{stderr_output}")
    
    process.stdout.close()
    process.stderr.close()
    process.wait()  # Wait for the process to finish

    end_time = time.time()  # End the timer
    
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f'{script} finished. Time taken: {elapsed_time:.2f} seconds.\n')