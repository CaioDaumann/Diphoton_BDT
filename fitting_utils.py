# we need to calculate (fit in the future) the simplified significance (s/sqrt(s+b)) and compare it to the mass resolution gain! 
#nescessary ones
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import glob
import os

import yaml
from yaml import Loader

import xgboost
from scipy.stats import chisquare
from sklearn.model_selection import train_test_split

#plotting libraries
import mplhep, hist
import mplhep as hep
plt.style.use([mplhep.style.CMS])
from matplotlib import pyplot