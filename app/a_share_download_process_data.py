# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# [1.] Download the a-share raw data.
# [2.] Manually add features to a-share data.
# [3.] Clean data. <a> get rid of nan feature value.
# [4.1] Data pre-processing and label the data. <a> PCA
# [4.2] label the data directly
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
from a_share import Ashare
# ==========================================================================================================

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>




# # ==========================================================================================================
# # [2.] Manually add features to a-share data.
# # ==========================================================================================================
# a_share1 = Ashare()
# input_folder = os.path.join(parent_folder, 'data', 'a_share', 'a_share_raw_data')
# save_folder = os.path.join(parent_folder, 'data', 'a_share', 'a_share_f_engineered_data')
# a_share1.feature_engineering(input_folder , save_folder)
# # ==========================================================================================================


# # ==========================================================================================================
# # [3.] Clean data. <a> get rid of nan feature value.
# # ==========================================================================================================
# data_cleaner = DataPp()
# input_folder = 'a_share_f_engineered_data'
# input_folder = os.path.join(parent_folder, 'data', 'a_share', input_folder)
# save_folder = 'a_share_processed_data'
# save_folder = os.path.join(parent_folder, 'data', 'a_share', save_folder)
# data_cleaner.correct_non_float_feature(input_folder, save_folder)
# #data_cleaner.examine_data(input_folder)  # examine the feature to see whether it is float
#
# # ==========================================================================================================


# # # ==========================================================================================================
# # # [4.] Label the data without PCA, etc.
# # # ==========================================================================================================
# a_share1 = Ashare()
# input_folder = os.path.join(parent_folder, 'data', 'a_share', 'a_share_processed_data')
# save_folder = os.path.join(parent_folder, 'data', 'a_share', 'a_share_labeled_data')
# a_share1.label_data(input_folder, save_folder)
# # # ==========================================================================================================


# # ==========================================================================================================
# # [4.1] PCA
# # ==========================================================================================================
stock_pca1 = StockPca(n_components = 10)
input_folder = os.path.join(parent_folder, 'data', 'a_share', 'a_share_labeled_data')
save_folder = os.path.join(parent_folder, 'data', 'a_share', 'a_share_labeled_data_[PCA]')
stock_pca1.transfrom_data_by_pca(input_folder, save_folder)
# # ==========================================================================================================