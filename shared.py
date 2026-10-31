'''
Oren Gershony
Shared constants and utils among classes
'''

import os

'''
SHARED CONSTANTS
'''

DATASET = "dataset"

TRAIN = "train"
TEST = "test"
VAL = "val"

TRAIN_PATH = os.path.join(DATASET, TRAIN)
TEST_PATH = os.path.join(DATASET, TEST)
VAL_PATH = os.path.join(DATASET, VAL)
