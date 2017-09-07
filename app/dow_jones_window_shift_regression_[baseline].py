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
from mlp_trade_regressor import MlpTradeRegressor
from trade_general_funcs import get_full_feature_switch_tuple
from trade_general_funcs import read_pca_component
# ==========================================================================================================

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# ==========================================================================================================
# Build MLP classifier
# ==========================================================================================================
print ("Build MLP classifier for dow jones data with window shift!")

# (1.) build classifer
mlp_classifier1 = MlpTradeRegressor()
#hidden_layer_sizes = (26,6)
#learning_rate_init = 0.0001
#print ("hidden_layer_sizes: ", hidden_layer_sizes)
##mlp_classifier1.set_regressor(hidden_layer_sizes, learning_rate_init = learning_rate_init)
#clsfy_name = 'dow_jones_mlp_trade_classifier_window_shift'
#clf_path = os.path.join(parent_folder, 'trained_classifiers', clsfy_name)

# (2.) data folder
data_folder = os.path.join('dow_jones_index_extended','dow_jones_index_extended_regression_test2')
data_folder = os.path.join(parent_folder, 'data', data_folder)
#

# (.) feature_switch
feature_switch_tuple_all_1 = get_full_feature_switch_tuple(data_folder)

# (3.) feed data
data_per = 1.0  # the percentage of data using for training and testing
shifting_size = 26
shift_num = 5
training_window_size = 74


pca_n_component = read_pca_component(data_folder)
mode = 'reg'
mlp_classifier1.trade_feed_and_separate_data_window_shift(data_folder, data_per=data_per, shifting_size=shifting_size,
                                                          feature_switch_tuple=feature_switch_tuple_all_1,mode = mode,
                                                          training_window_size =training_window_size, shift_num = shift_num,
                                     is_standardisation = False, is_PCA = False, pca_n_component = pca_n_component)

# (4.) train window shift
random_seed = 'window_shift'
average_f1_list = []
accuracy_list = []
rmse_list = []
avg_pc_list = []

is_baseline = True
for shift in mlp_classifier1.validation_dict[random_seed].keys():
    mlp_classifier1.trade_rs_cv_load_train_dev_data(random_seed, shift)
    week_average_rmse, avg_price_change_tuple, week_average_accuracy, week_average_f1, date_actual_avg_priceChange_list = \
        mlp_classifier1.baseline_reg_dev(target_folder = data_folder)
    average_f1_list.append(week_average_f1)
    accuracy_list.append(week_average_accuracy)
    rmse_list.append(week_average_rmse)
    avg_pc_list.append(avg_price_change_tuple)


print ("is baseline: {}".format(is_baseline))
print ("average f1 over window shifting: {}".format(np.average(average_f1_list)))
print ("average accuracy over window shifting: {}".format(np.average(accuracy_list)))
print ("average rmse over window shifting: {}".format(np.average(rmse_list)))
print ("average average price over window shifting: {}".format(np.average(avg_pc_list)))
print ("date_actual_avg_priceChange_list: ", len(date_actual_avg_priceChange_list))
# # ==========================================================================================================

# ----------------------------------------------------------------------------------------------------------
# BASELINE
# ----------------------------------------------------------------------------------------------------------
# average f1 over window shifting: 0.5282581617125446
# average accuracy over window shifting: 0.5566666666666668
# ----------------------------------------------------------------------------------------------------------
