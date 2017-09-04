# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Build the MLP regressor.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# (c) 2017 PJS, University of Sheffield, iamlxb3@gmail.com
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# ==========================================================================================================
# general package import
# ==========================================================================================================
import sys
import os
import random
import collections
import numpy as np
# ==========================================================================================================

# ==========================================================================================================
# ADD SYS PATH
# ==========================================================================================================
parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
clf_path = os.path.join(parent_folder, 'classifiers', 'mlp')
path2 = os.path.join(parent_folder, 'general_functions')

sys.path.append(clf_path)
sys.path.append(path2)
# ==========================================================================================================

# ==========================================================================================================
# local package import
# ==========================================================================================================
from mlp_trade_regressor import MlpTradeRegressor
from trade_general_funcs import compute_f1_accuracy, compute_trade_weekly_clf_result, get_chosen_stock_return, \
    plot_stock_return, calculate_rmse
from trade_general_funcs import get_avg_price_change
# ==========================================================================================================

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# ------------------------------------------------------------------------------------------------------------
# read the dataset
# ------------------------------------------------------------------------------------------------------------
# train
train_data_folder = os.path.join('dow_jones_index_extended','dow_jones_index_extended_regression')
train_data_folder = os.path.join(parent_folder, 'data', train_data_folder)
# test
test_data_folder = os.path.join('dow_jones_index_extended','dow_jones_index_extended_regression_test')
test_data_folder = os.path.join(parent_folder, 'data', test_data_folder)
# ------------------------------------------------------------------------------------------------------------



# ==========================================================================================================
# Build MLP classifier for a-share data, save the mlp to local
# ==========================================================================================================
print ("Build MLP regressor for dow_jones extended test data!")
mode = 'reg' #'reg'


# ------------------------------------------------------------------------------------------------------------
# hyper parameters for DATA
# ------------------------------------------------------------------------------------------------------------
is_standardisation = True
is_PCA = True
pca_n_component = 10
# ------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------
# hyper parameters for ANNs
# ------------------------------------------------------------------------------------------------------------
verbose = False
hidden_layer_sizes = (33,3)
tol = 1e-8
learning_rate_init = 0.001
learning_rate = 'constant'
early_stopping = False
activation  = 'relu'
validation_fraction  = 0.1 # The proportion of training data to set aside as validation set for early stopping.
                           # Must be between 0 and 1. Only used if early_stopping is True.
alpha  = 0.0001
# ------------------------------------------------------------------------------------------------------------
RANDOM_STATE_TEST_NUM = 10
random_state_pool = [random.randint(0,99999) for x in range(0,RANDOM_STATE_TEST_NUM)]
best_f1_list = [0,0,0] # f1, accuracy, random_state
best_accuracy_list = [0,0,0] # accuracy, f1, random_state
best_random_state = -1
best_rmse_list = [float('inf'),0,0]
best_avg_pc_list = [0,0,0]
best_return_list = []
best_date_list = []
# ------------------------------------------------------------------------------------------------------------


for random_state in random_state_pool:

    # (1.) build classifer
    mlp1 = MlpTradeRegressor()
    clsfy_name = 'dow_jones_extened_test_mlp_regressor'
    clf_path = os.path.join(parent_folder, 'trained_classifiers', clsfy_name)
    data_per = 1.0 # the percentage of data using for training and testing
    #

    # (2.) read window_size and other config
    is_moving_window = True
    week_for_predict = 50  # None
    # window size
    _1, _2, date_str_list, _3 = mlp1._feed_data(test_data_folder, data_per, feature_switch_tuple=None,
                                                is_random=False, random_seed=1, mode=mode)
    date_str_set = set(date_str_list)
    test_date_num = len(date_str_set)
    if is_moving_window:
        window_size = 1
    else:
        window_size = test_date_num
    #
    # window index
    window_index_start = 0
    wasted_date_num = test_date_num % window_size
    if wasted_date_num != 0:
        print ("Some instances in test set may be wasted!! test_date_num%window_size: {}".format(wasted_date_num))
    max_window_index = int(test_date_num/window_size)
    #


    pred_label_list = []
    actual_label_list = []
    data_list_for_classification = []
    avg_price_change_list = []
    rmse_list = []
    chosen_stock_return_list = []
    plot_date_list = []
    var_list = []
    std_list = []



    for window_index in range(window_index_start, max_window_index):
        print ("===window_index: {}===".format(window_index))
        # (2.) load training data, save standardisation_file and pca_file
        standardisation_file_path = os.path.join(parent_folder, 'data_processor','z_score')
        pca_file_path = os.path.join(parent_folder,'data_processor','pca')
        mlp1.trade_feed_and_separate_data_for_test(train_data_folder, test_data_folder, data_per = data_per,
                                          standardisation_file_path = standardisation_file_path,
                                          pca_file_path = pca_file_path, mode = mode, pca_n_component = pca_n_component
                                          ,is_standardisation = is_standardisation, is_PCA = is_PCA,
                                                   is_moving_window = is_moving_window, window_size = window_size,
                                                   window_index = window_index, week_for_predict = week_for_predict)

        # (3.) load hyper parameters and training
        mlp1.set_regressor(hidden_layer_sizes, tol=tol, learning_rate_init=learning_rate_init, random_state=random_state,
                                   verbose = verbose, learning_rate = learning_rate, early_stopping =early_stopping,
                                    activation  = activation, validation_fraction  = validation_fraction, alpha  = alpha)
        mlp1.regressor_train(save_clsfy_path= clf_path)
        pred_value_list, actual_value_list, date_list, stock_id_list = mlp1.reg_dev_for_moving_window_test(save_clsfy_path= clf_path)
        #
        pred_label_list_temp = ['pos' if x >= 0 else 'neg' for x in pred_value_list]
        actual_label_list_temp = ['pos' if x >= 0 else 'neg' for x in actual_value_list]
        pred_label_list.extend(pred_label_list_temp)
        actual_label_list.extend(actual_label_list_temp)
        data_list_for_classification.extend(date_list)
        #
        rmse = calculate_rmse(actual_value_list, pred_value_list)
        avg_price_change_tuple, var_tuple, std_tuple = get_avg_price_change(pred_value_list, actual_value_list
                                                                            , date_list, stock_id_list)

        chosen_stock_return_list_temp = get_chosen_stock_return(pred_value_list, actual_value_list, date_list, stock_id_list)
        avg_price_change_1 = avg_price_change_tuple[0] # Strategy: choose the top 1 stock each week
        var_1 = var_tuple[0]
        std_1 = std_tuple[0]
        avg_price_change_list.append(avg_price_change_1)
        var_list.append(var_1)
        std_list.append(std_1)
        chosen_stock_return_list.extend(chosen_stock_return_list_temp)
        rmse_list.append(rmse)


        # date set
        date_set = set(date_list)
        sorted_date_list = sorted(list(date_set))
        plot_date_list.extend(sorted_date_list)
        #

        print ("Regressor for test trained successfully!")

    week_average_f1, week_average_accuracy, dev_label_dict, pred_label_dict = \
        compute_trade_weekly_clf_result(pred_label_list, actual_label_list, data_list_for_classification)

    avg_price_change = np.average(avg_price_change_list)
    avg_std = np.average(std_list)
    avg_rmse = np.average(rmse_list)

    print ("---------------------------------------------------")
    print ("random_state: ", random_state)
    print ("avg_rmse: ", avg_rmse)
    print ("avg_f1: ", week_average_f1)
    print ("accuracy: ", week_average_accuracy)
    print ("avg_price_change: ", avg_price_change)
    print ("avg_std: ", avg_std)
    print ("window_index_start: {}, max_window_index: {}, window_size: {}".format(window_index_start,max_window_index,
                                                                                  window_size))
    print ("pred_label_dict: {}".format(pred_label_dict.items()))
    print ("dev_label_dict: {}".format(dev_label_dict.items()))

    #
    if avg_rmse < best_rmse_list[0]:
        best_rmse_list[0] = avg_rmse
        best_rmse_list[1] = avg_price_change
        best_rmse_list[2] = random_state

    if avg_price_change > best_avg_pc_list[0]:
        best_avg_pc_list[0] = avg_price_change
        best_avg_pc_list[1] = avg_rmse
        best_avg_pc_list[2] = random_state
        best_return_list = chosen_stock_return_list # to plot stock returns with best avg_price
    #
    best_date_list = plot_date_list


print ("+++++++++++++++++++++++++++++++++++++++++++")
print ("best_rmse: {}, avg_pc: {}, random_state: {}".format(*best_rmse_list))
print ("best_avg_pc: {}, rmse: {}, random_state: {}".format(*best_avg_pc_list))
print ("+++++++++++++++++++++++++++++++++++++++++++")


# ----------------------------------------------------------------------------------------------------------------------
# plot for the stock return
capital = 1
title = 'Stock return for MLP regressor'
xlabel ='Date'
file_name = "{}.png".format('Regressor_stock_return')
save_path = os.path.join(parent_folder, 'results', file_name)
plot_stock_return(best_return_list, best_date_list, capital = capital,
                  title = title, xlabel = xlabel, save_path = save_path)

# ----------------------------------------------------------------------------------------------------------------------
