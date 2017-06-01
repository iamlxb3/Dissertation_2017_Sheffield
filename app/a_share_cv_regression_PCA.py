# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# The complete CV test for a share MLP regressor
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# (c) 2017 PJS, University of Sheffield, iamlxb3@gmail.com
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# ==========================================================================================================
# general package import
# ==========================================================================================================
import sys
import os
import itertools
import numpy as np
# ==========================================================================================================

# ==========================================================================================================
# ADD SYS PATH
# ==========================================================================================================
parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mlp_path = os.path.join(parent_folder, 'classifiers','mlp')
sys.path.append(mlp_path)
# ==========================================================================================================

# ==========================================================================================================
# local package import
# ==========================================================================================================
from mlp_classifier import MlpClassifier
# ==========================================================================================================

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# ==========================================================================================================
# Build MLP classifier for a-share data, save the mlp to local
# ==========================================================================================================
# (1.) build classifer
mlp_regressor1 = MlpClassifier()


# a share data
data_folder = os.path.join('a_share','a_share_regression_PCA_data')
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

# (2.) clf_path
clsfy_name = 'a_share_mlp_cv_PCA_regressor'
clf_path = os.path.join(parent_folder, 'trained_classifiers', clsfy_name)

# ----------------------------------------------------------------------------------------------------------------------


# TEST ONE FEATURE combination
feature_switch_tuple = (1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
print("========================================================================================")
print("PCA 10 fold cross validation testing start!! feature_switch_tuple: {}".format(feature_switch_tuple))
print("========================================================================================")



# ==========================================================================================================
# Build MLP classifier for a-share data, save the mlp to local
# ==========================================================================================================
# (1.) build classifer
hidden_layer_sizes = (36, 6)
learning_rate_init = 0.0001
mlp_regressor1.set_regressor(hidden_layer_sizes, learning_rate_init = learning_rate_init)
include_top = 1
random_seed = 99
tol = 1e-6

print("Hidden layer size: {}".format(hidden_layer_sizes))
print("strategy: include_top={}".format(include_top))

samples_feature_list, samples_value_list, \
date_str_list, stock_id_list = mlp_regressor1._r_feed_data(data_folder, data_per = 1.0,
                                                 feature_switch_tuple=feature_switch_tuple, is_random=True,
                                                           random_seed = random_seed)


# (b.) 10-cross-validation train and test
for validation_index in range(10):
    mlp_regressor1.cv_r_feed_data_train_test(validation_index, samples_feature_list, samples_value_list,
                                   date_str_list, stock_id_list)
    mlp_regressor1.regressor_train(save_clsfy_path=clf_path, is_cv=True)
    mlp_regressor1.regressor_dev(save_clsfy_path=clf_path, is_cv=True, include_top=include_top)


# (d.) real-time print
print("====================================================================")
print("Feature selected: {}, Total number: {}".format(feature_switch_tuple,
                                                      len(feature_switch_tuple)))
print("Average mres: {}".format(np.average(mlp_regressor1.mres_list)))
print("Average price change: {}".format(np.average(mlp_regressor1.avg_price_change_list)))
print("Average var: {}, Average std: {}".format(np.average([x[0] for x in mlp_regressor1.var_std_list]),
                                                np.average([x[1] for x in mlp_regressor1.var_std_list])))
print("Average polarity: {}".format(np.average(mlp_regressor1.polar_accuracy_list)))
print("Average iteration_loss: {}".format(np.average([x[1] for x in mlp_regressor1.iteration_loss_list])))
print("====================================================================")
print("mres: {}".format(mlp_regressor1.mres_list))
print("price change: {}".format(mlp_regressor1.avg_price_change_list))
print("var: {} std: {}".format([x[0] for x in mlp_regressor1.var_std_list], [x[1] for x in mlp_regressor1.var_std_list]))
print("polarity: {}".format(mlp_regressor1.polar_accuracy_list))
print("iteration_loss: {}".format([x[1] for x in mlp_regressor1.iteration_loss_list]))
print("====================================================================")
# ==========================================================================================================

