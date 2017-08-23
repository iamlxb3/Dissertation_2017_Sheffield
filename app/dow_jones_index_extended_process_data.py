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
from dow_jones_extended import DowJonesIndexExtended
# ==========================================================================================================

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# # ==========================================================================================================
# # [1.] format raw data
# # ==========================================================================================================
# dow_jones_index1 = DowJonesIndexExtended()
# input_folder = os.path.join(parent_folder, 'data', 'dow_jones_index_extended', 'dow_jones_index_extended_original')
# save_folder = os.path.join(parent_folder, 'data', 'dow_jones_index_extended', 'dow_jones_index_extended_raw')
# dow_jones_index1.format_raw_data(input_folder, save_folder)
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
dow_jones_index1 = DowJonesIndexExtended()
input_file = os.path.join(parent_folder, 'data', 'dow_jones_index_extended', 'dow_jones_index_extended_raw')
save_folder = os.path.join(parent_folder, 'data', 'dow_jones_index_extended', 'dow_jones_index_extended_f_engineered')
dow_jones_index1.feature_engineering(input_file, save_folder)
# ==========================================================================================================



# # ==========================================================================================================
# # [3.1] feature engineering, add more data (previous week)
# # ==========================================================================================================
# dow_jones_index1 = DowJonesIndexExtended()
# input_file = os.path.join(parent_folder, 'data', 'dow_jones_index_extended', 'dow_jones_index_extended_f_engineered')
# save_folder = os.path.join(parent_folder, 'data', 'dow_jones_index_extended', 'dow_jones_index_extended_f_engineered_more_data')
# dow_jones_index1.f_engineering_add_1_week_data(input_file, save_folder)
# # ==========================================================================================================

# # # ==========================================================================================================
# # # [4.] scaling, z-score
# # # ==========================================================================================================
# data_cleaner = DataPp()
# input_folder = 'dow_jones_index_f_engineered'
# input_folder = os.path.join(parent_folder, 'data', 'dow_jones_index', input_folder)
# save_folder = 'dow_jones_scaled_data'
# save_folder = os.path.join(parent_folder, 'data', 'dow_jones_index', save_folder)
#
#
# features_scale_list = ''
#
# # scale data and save scaler
# trained_classifiers_folder = os.path.join(parent_folder, 'trained_data_processor')
# scaler_name = 'a_share_z_score_scaler'
#
# data_cleaner.scale_data(input_folder, save_folder,  features_scale_list,
#                         trained_classifiers_folder, scaler_name, mode = 'z_score', data_set = 'dow_jones')
# # # ==========================================================================================================


# ==========================================================================================================
# [5.] data labeling
# ==========================================================================================================
dow_jones_index1 = DowJonesIndexExtended()
input_file = os.path.join(parent_folder, 'data', 'dow_jones_index_extended', 'dow_jones_index_extended_f_engineered_more_data')
save_folder = os.path.join(parent_folder, 'data', 'dow_jones_index_extended', 'dow_jones_index_extended_labeled')
dow_jones_index1.label_data(input_file, save_folder)
# ==========================================================================================================

# # # ==========================================================================================================
# # # PCA-clf
# # # ==========================================================================================================
# stock_pca1 = StockPca(n_components = 12)
# input_folder = os.path.join(parent_folder, 'data', 'dow_jones_index', 'dow_jones_index_labeled')
# save_folder = os.path.join(parent_folder, 'data', 'dow_jones_index', 'dow_jones_index_labeled_PCA')
# stock_pca1.transfrom_data_by_pca(input_folder, save_folder)
# # # ==========================================================================================================


# ==========================================================================================================
# [5.] data regression
# ==========================================================================================================
dow_jones_index1 = DowJonesIndexExtended()
input_folder = os.path.join(parent_folder, 'data', 'dow_jones_index_extended', 'dow_jones_index_extended_f_engineered_more_data')
save_folder = os.path.join(parent_folder, 'data', 'dow_jones_index_extended', 'dow_jones_index_extended_regression')
dow_jones_index1.price_change_regression(input_folder, save_folder)
# ==========================================================================================================

# # # ==========================================================================================================
# # # PCA-regression
# # # ==========================================================================================================
# stock_pca1 = StockPca(n_components = 12)
# input_folder = os.path.join(parent_folder, 'data', 'dow_jones_index', 'dow_jones_index_regression')
# save_folder = os.path.join(parent_folder, 'data', 'dow_jones_index', 'dow_jones_index_regression_PCA')
# stock_pca1.transfrom_data_by_pca(input_folder, save_folder)
# # # ==========================================================================================================



