import pandas as pd

# Load the datasets
true_positives_path = '/mnt/data/CADDY_gestures_all_true_positives_release_v2.csv'
true_negatives_path = '/mnt/data/CADDY_gestures_all_true_negatives_release_v2.csv'

true_positives = pd.read_csv(true_positives_path)
true_negatives = pd.read_csv(true_negatives_path)

# Inspect the first few rows of each dataset to understand their structure
true_positives_head = true_positives.head()
true_negatives_head = true_negatives.head()

# Get a summary of each dataset
true_positives_info = true_positives.info()
true_negatives_info = true_negatives.info()

true_positives_head, true_negatives_head, true_positives_info, true_negatives_info

import sys

# Open a log file
with open('logfile.txt', 'w') as f:
    # Redirect stdout to the log file
    sys.stdout = f
    # Now, print statements will go to 'logfile.txt' instead of the console
    print("This will be written to the log file.")

# Reset stdout to default to restore console output
sys.stdout = sys.__stdout__



import logging

# Configure logging
logging.basicConfig(filename='logfile.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Example logging
logging.info('Starting preprocessing')


