# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# The complete PCA CV test for a share MLP regressor
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# (c) 2017 PJS, University of Sheffield, iamlxb3@gmail.com
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# ==========================================================================================================
# general package import
# ==========================================================================================================
import sys
import os
import numpy as np
# ==========================================================================================================

# ==========================================================================================================
# ADD SYS PATH
# ==========================================================================================================
parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mlp_path = os.path.join(parent_folder, 'classifiers', 'mlp')
path2 = os.path.join(parent_folder, 'general_functions')
sys.path.append(mlp_path)
# ==========================================================================================================

# ==========================================================================================================
# local package import
# ==========================================================================================================
from mlp_trade_classifier import MlpTradeClassifier
from trade_general_funcs import get_full_feature_switch_tuple
from trade_general_funcs import read_pca_component
# ==========================================================================================================

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



# (1.) build classifer
mlp_classifier1 = MlpTradeClassifier()

# (2.) data folder
data_folder = os.path.join('dow_jones_index','dow_jones_index_labeled')
data_folder = os.path.join(parent_folder, 'data', data_folder)


data_image_folder = os.path.join(parent_folder, 'data_image', 'dow_jones_index')
#

# (.) feature_switch
feature_switch_tuple_all_1 = get_full_feature_switch_tuple(data_folder)

# (3.) feed data
data_per = 1.0  # the percentage of data using for training and testing
shifting_size_percent = 0.1
shift_num = 5
pca_n_component = read_pca_component(data_folder)
mode = 'clf'
mlp_classifier1.trade_feed_and_separate_data_window_shift(data_folder, data_per=data_per,
                                                          feature_switch_tuple=feature_switch_tuple_all_1,mode = mode,
                                                  shifting_size_percent =shifting_size_percent, shift_num = shift_num,
                                     is_standardisation = True, is_PCA = True, pca_n_component = pca_n_component)

# (4.) save data image
random_seed = 'window_shift'
for shift,_ in mlp_classifier1.validation_dict[random_seed].items():
    training_feature_set = mlp_classifier1.validation_dict[random_seed][shift]['training_set']
    training_value_set = mlp_classifier1.validation_dict[random_seed][shift]['training_value_set']
    title = "DowJonesData_window_shift-{}".format(shift)
    save_path = os.path.join(parent_folder, 'data_image', 'dow_jones_index', title + '.png')
    mlp_classifier1.save_data_image_PCA(training_feature_set, training_value_set, title, save_path)
