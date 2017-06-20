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
# Build MLP regressor for a-share data, save the mlp to local
# ==========================================================================================================
# (1.) build classifer
mlp_regressor1 = MlpTradeRegressor()

# # TODO ADD SOME regressor TESTING DATA
# data_folder = os.path.join('dow_jones','dow_jones_regressor_PCA_data')
# data_folder = os.path.join(parent_folder, 'data', data_folder)
# #



# a share data
data_folder = os.path.join('dow_jones_index','dow_jones_index_regression')
#data_folder = os.path.join('a_share','a_share_regression_PCA_data')
data_folder = os.path.join(parent_folder, 'data', data_folder)
#

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
# FEATURE AND TOPOLOGY TEST
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
# config the other parameters
# ----------------------------------------------------------------------------------------------------------------------
other_config_dict = {}
# (1.) mlp config
other_config_dict['data_per'] = 1.0
other_config_dict['learning_rate_init'] = 0.0001
other_config_dict['tol'] = 1e-6
# clsfy
clsfy_name = 'dow_jones_mlp_cv_PCA_regressor_window_shift'
other_config_dict['clf_path'] = os.path.join(parent_folder, 'trained_classifiers', clsfy_name)

# (2.) data mode config
other_config_dict['shifting_size_percent'] = 0.1
other_config_dict['shift_num'] = 5

# (3.) trading strategy config
other_config_dict['include_top_list'] = [1]

# (4.) data pre-processing config
other_config_dict['is_standardisation'] = True
other_config_dict['is_PCA'] = True
other_config_dict['is_PCA_feature_degradation'] = True
pca_n_component = read_pca_component(data_folder)
pca_n_component_list = sorted([i for i in range(pca_n_component + 1) if i != 0 ], reverse = True)


# (3.) topology_result_path
rmse_result_path = os.path.join(parent_folder, 'topology_feature_test',
                                       'dow_jones_cv_regressor_PCA_rmse_window_shift.txt')
polarity_result_path = os.path.join(parent_folder, 'topology_feature_test',
                                       'dow_jones_cv_regressor_PCA_polarity_window_shift.txt')
avg_pc_result_path = os.path.join(parent_folder, 'topology_feature_test',
                                       'dow_jones_cv_regressor_PCA_avg_pc_window_shift.txt')
# ----------------------------------------------------------------------------------------------------------------------



# ----------------------------------------------------------------------------------------------------------------------
# decreasing the number of features for PCA 1 by 1
# ----------------------------------------------------------------------------------------------------------------------
# get_full_feature_switch_tuple, (1,1,1,1,1,1,....)
feature_switch_tuple_all_1 = get_full_feature_switch_tuple(data_folder)

# ALL FEATURES
feature_switch_tuple_list = [feature_switch_tuple_all_1]
#

print ("feature_switch_tuple_list: ", feature_switch_tuple_list)
# ----------------------------------------------------------------------------------------------------------------------

# TEST ONE FEATURE combination
#feature_switch_tuple_list = [(1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)]
print("====================================================================")
print("PCA feature test with window shift start!! Total feature combination: {}".format(len(feature_switch_tuple_list)))
print("====================================================================")


# testing the performance of regressor under different number of PCA features
for feature_switch_tuple in feature_switch_tuple_list:
    mlp_regressor1.read_selected_feature_list(data_folder, feature_switch_tuple)

    # PCA degradation
    if other_config_dict['is_PCA_feature_degradation']:
        print ("PCA_feature_degradation is on!")
        print ("PCA_n_component_list: {}".format(pca_n_component_list))
        for pca_n_component in pca_n_component_list:
            print("-----------------------------------------------------------------")
            print ("PCA n_component now: ", pca_n_component)
            print("-----------------------------------------------------------------")
            other_config_dict['pca_n_component'] = pca_n_component
            # ----------------------------------------------------------------------------------------------------------------------

            # ----------------------------------------------------------------------------------------------------------------------
            # config hidden layer size
            # ----------------------------------------------------------------------------------------------------------------------
            hidden_layer_node_min = 20
            hidden_layer_node_max = 200
            hidden_layer_node_step = 1
            hidden_layer_depth_min = 3
            hidden_layer_depth_max = 10

            hidden_layer_node_min = 1
            hidden_layer_node_max = 2
            hidden_layer_node_step = 1
            hidden_layer_depth_min = 1
            hidden_layer_depth_max = 2

            hidden_layer_config_tuple = (hidden_layer_node_min, hidden_layer_node_max, hidden_layer_node_step, hidden_layer_depth_min,
                                         hidden_layer_depth_max)
            # ----------------------------------------------------------------------------------------------------------------------
            print("====================================================================")
            print("Start testing for MLP's topology for regression for dow jones data set!")
            print("====================================================================")

            # run topology test, window shift
            mlp_regressor1.cv_r_topology_test(data_folder, feature_switch_tuple, other_config_dict, hidden_layer_config_tuple
                                                 , is_window_shift = True)
            # ======================================================================================================================

        mlp_regressor1.cv_r_save_feature_topology_result(rmse_result_path, key = 'rmse')
        mlp_regressor1.cv_r_save_feature_topology_result(avg_pc_result_path, key = 'avg_pc')
        mlp_regressor1.cv_r_save_feature_topology_result(polarity_result_path, key = 'polar')
        # ==========================================================================================================
        # ==========================================================================================================
        # ==========================================================================================================

