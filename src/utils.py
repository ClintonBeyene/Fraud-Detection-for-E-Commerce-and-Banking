import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import struct
import socket
import logging 
import os 
import sys

# Add the parent directory to the system path
sys.path.append(os.path.join(os.path.abspath('../')))

# Ensure the logs directory exists
log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../logs"))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Set up logging
logging.basicConfig(
    filename=os.path.join(log_dir, 'utils.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def ip_to_int(ip):
    try:
        return struct.unpack("!I", socket.inet_aton(ip))[0]
    except struct.error:
        return None


