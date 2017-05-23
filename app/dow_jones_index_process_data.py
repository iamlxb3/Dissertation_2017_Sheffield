# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# [1.] format raw data
# [2.] handle missing data
# [3.] feature engineering
# [4.] clean the data
# [5.] data transformation eg.PCA
# [6.] data labeling

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# (c) 2017 PJS, University of Sheffield, iamlxb3@gmail.com
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# ==========================================================================================================
# general package import
# ==========================================================================================================
import sys
import os
# ==========================================================================================================

# ==========================================================================================================
# ADD SYS PATH
# ==========================================================================================================
parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_processor_path = os.path.join(parent_folder, 'data_processor')
data_generator_path = os.path.join(parent_folder, 'data_generator')
sys.path.append(data_processor_path)
sys.path.append(data_generator_path)
# ==========================================================================================================

# ==========================================================================================================
# local package import
# ==========================================================================================================
from data_preprocessing import DataPp
from stock_pca import StockPca
from dow_jones_index import DowJonesIndex
# ==========================================================================================================

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# # ==========================================================================================================
# # [1.] format raw data
# # ==========================================================================================================
# dow_jones_index1 = DowJonesIndex()
# input_file = os.path.join(parent_folder, 'data', 'dow_jones_index', 'dow_jones_index_original',
#                           'dow_jones_index.data')
# save_folder = os.path.join(parent_folder, 'data', 'dow_jones_index', 'dow_jones_index_raw')
# dow_jones_index1.format_raw_data(input_file, save_folder)
# # ==========================================================================================================


# # ==========================================================================================================
# # [2.] fill in nan data
# # ==========================================================================================================
# dp1 = DataPp()
# input_file = os.path.join(parent_folder, 'data', 'dow_jones_index','dow_jones_index_raw')
# save_folder = os.path.join(parent_folder, 'data', 'dow_jones_index', 'dow_jones_index_fill_nan')
# dp1.fill_in_nan_data(input_file, save_folder)
# # ==========================================================================================================


# ==========================================================================================================
# [3.] feature engineering
# ==========================================================================================================
dow_jones_index1 = DowJonesIndex()
input_file = os.path.join(parent_folder, 'data', 'dow_jones_index', 'dow_jones_index_fill_nan')
save_folder = os.path.join(parent_folder, 'data', 'dow_jones_index', 'dow_jones_index_f_engineered')
dow_jones_index1.feature_engineering(input_file, save_folder)
# ==========================================================================================================


# ==========================================================================================================
# [6.] data labeling
# ==========================================================================================================
dow_jones_index1 = DowJonesIndex()
input_file = os.path.join(parent_folder, 'data', 'dow_jones_index', 'dow_jones_index_f_engineered')
save_folder = os.path.join(parent_folder, 'data', 'dow_jones_index', 'dow_jones_index_labeled')
dow_jones_index1.label_data(input_file, save_folder)
# ==========================================================================================================