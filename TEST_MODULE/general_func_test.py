# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# unit test on general funcs
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# (c) 2017 PJS, University of Sheffield, iamlxb3@gmail.com
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# ==========================================================================================================
# general package import
# ==========================================================================================================
import sys
import math
import numpy as np
import os
from sklearn.metrics import mean_squared_error
# ==========================================================================================================

# ==========================================================================================================
# ADD SYS PATH
# ==========================================================================================================
parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mlp_path = os.path.join(parent_folder, 'general_functions')
sys.path.append(mlp_path)
# ==========================================================================================================

# ==========================================================================================================
# local package import
# ==========================================================================================================
from general_funcs import calculate_mrse
# ==========================================================================================================


# ==========================================================================================================
# TEST-[1] <<<RMSE>>>
# ==========================================================================================================

# TODO complete the unit test
y_actual = np.array([1,2,3.2,8.7,5.5,112])
y_predicted= np.array([2,3.9,4.1,0,34.3,9.9])

rms1 = math.sqrt(mean_squared_error(y_actual, y_predicted))
rms2 = calculate_mrse(y_actual, y_predicted)

print ("rms1: {}".format(rms1))
print ("rms2: {}".format(rms2))

# ==========================================================================================================


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>