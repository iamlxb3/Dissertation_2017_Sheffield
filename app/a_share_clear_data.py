# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# clear all the a share data
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

raw = os.path.join(parent_folder, 'data', 'a_share', 'a_share_raw_data')
f_engineered = os.path.join(parent_folder, 'data', 'a_share', 'a_share_f_engineered_data')
processed = os.path.join(parent_folder, 'data', 'a_share', 'a_share_processed_data')
labeled = os.path.join(parent_folder, 'data', 'a_share', 'a_share_labeled_data')
regression = os.path.join(parent_folder, 'data', 'a_share', 'a_share_regression_data')
scaled = os.path.join(parent_folder, 'data', 'a_share', 'a_share_scaled_data')
PCA = os.path.join(parent_folder, 'data', 'a_share', 'a_share_regression_PCA_data')


delete_folder_list = [raw, f_engineered, processed, labeled, regression, scaled, PCA]
delete_folder_list = [f_engineered]

for folder in delete_folder_list:
    remove_count = 0
    delete_file_names = os.listdir(folder)
    for file_name in delete_file_names:
        file_path = os.path.join(folder, file_name)
        os.remove(file_path)
        remove_count += 1
    print ("Totally remove {} files in folder {}".format(remove_count, os.path.basename(folder)))

