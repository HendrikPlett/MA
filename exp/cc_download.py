"""
Used to download the desired CausalChamber data into the needed cc_data folder.
"""

import os
from causalchamber.datasets import Dataset
from cc_wrapper import CHAMBER_CONFIGURATIONS

DATA_FOLDER = "cc_data"

def download(exp_family: str):
    if exp_family not in tuple(CHAMBER_CONFIGURATIONS.keys()):
            raise ValueError("Not a valid experimental family")
    path = os.path.join(os.path.dirname(__file__), DATA_FOLDER, exp_family)
    if not os.path.exists(path):
        Dataset(name=exp_family, 
                root=os.path.join(os.path.dirname(__file__), DATA_FOLDER), 
                download=True)
        print(f"Downloaded dataset {exp_family} under {path}")
    else:
        print(f"Dataset {exp_family} already exists under {path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Error! Usage: python3 cc_download.py <exp_family>")
        sys.exit(1)
    exp_family = sys.argv[1]
    download(exp_family)

