import os

host = os.uname()[1]

if host == 'nsk1compviz':
    HOME = '/home/de568294'
    PROJECTS_DIR = os.path.join(HOME, 'Projects')
    DATA_DIR = os.path.join(PROJECTS_DIR, 'data', 'symbolic')
    RESULTS_DIR = os.path.join(PROJECTS_DIR, 'results', 'symbolic', 'EL')

elif host == 'GC02X84DTJG5ME':
    HOME = '/Users/aritrachowdhury'
    PROJECTS_DIR = os.path.join(HOME, 'Projects')
    DATA_DIR = os.path.join(PROJECTS_DIR, 'data', 'symbolic')
    RESULTS_DIR = os.path.join(PROJECTS_DIR, 'results', 'symbolic', 'EL')

