# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Build the MLP regressor.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# (c) 2017 PJS, University of Sheffield, iamlxb3@gmail.com
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# ==========================================================================================================
# general package import
# ==========================================================================================================
import sys
import random
import os
import itertools
import numpy as np
import collections
# ==========================================================================================================

# ==========================================================================================================
# ADD SYS PATH
# ==========================================================================================================
parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path1 = os.path.join(parent_folder, 'general_functions')
sys.path.append(path1)
# ==========================================================================================================

# ==========================================================================================================
# local package import
# ==========================================================================================================
from stock_box_plot2 import model_result_box_plot
# ==========================================================================================================

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# ----------------------------------------------------------------------------------------------------------------------
# read data into dictionary
# ----------------------------------------------------------------------------------------------------------------------
data_set = 'dow_jones_index'
mode = 'clf'
classification_list = ['classifier','bagging_classifier','regressor','bagging_regressor','adaboost_regressor']
regression_list = ['regressor','bagging_regressor','adaboost_regressor']

classifier_list = ['classifier','bagging_classifier']
regressor_list = ['regressor','bagging_regressor','adaboost_regressor']
data_preprocessing_list = ['pca','pca_standardization','standardization','origin']


# input path

if mode =='clf':
    model_list = classification_list
elif mode == 'reg':
    model_list = regression_list

result_dict = collections.defaultdict(lambda :collections.defaultdict(lambda:collections.defaultdict(lambda:[])))


for model, data_preprocessing in list(itertools.product(model_list, data_preprocessing_list)):
    hyper_parameter_folder = os.path.join(parent_folder, 'hyper_parameter_test', data_set,
                                          model, data_preprocessing)
    file_name_list = os.listdir(hyper_parameter_folder)
    file_path_list = [os.path.join(hyper_parameter_folder, x) for x in file_name_list]

    print ("classifier: {}, data_preprocessing: {}".format(model, data_preprocessing))
    print ("file_path_list_length: ", len(file_path_list))

# save path


    # ----------------------------------------------------------------------------------------------------------------------
    # (1.) push every model's result to dict
    # ----------------------------------------------------------------------------------------------------------------------
    for i, file_path in enumerate(file_path_list):
        with open (file_path, 'r') as f:
            file_name = file_name_list[i]
            if file_name == '.gitignore' or  file_name == 'feature_selection.txt':
                continue
            for j, line in enumerate(f):
                unique_id = "{}_{}".format(i,j)
                line_list = line.strip().split(',')
                if model in classifier_list:
                    loss = line_list[0]
                    avg_iter_num = line_list[1]
                    avg_f1 = line_list[2]
                    accuracy = line_list[3]
                    result_dict[model][data_preprocessing]['loss_list'].append(float(loss))
                    result_dict[model][data_preprocessing]['avg_iter_num_list'].append(float(avg_iter_num))
                    result_dict[model][data_preprocessing]['avg_f1_list'].append(float(avg_f1))
                    result_dict[model][data_preprocessing]['accuracy_list'].append(float(accuracy))
                elif model in regressor_list:
                    loss = line_list[0]
                    avg_iter_num = line_list[1]
                    rmse = line_list[2]
                    avg_pc = line_list[3]
                    accuracy = line_list[4]
                    avg_f1 = line_list[5]
                    result_dict[model][data_preprocessing]['loss_list'].append(float(loss))
                    result_dict[model][data_preprocessing]['avg_iter_num_list'].append(float(avg_iter_num))
                    result_dict[model][data_preprocessing]['rmse_list'].append(float(rmse))
                    result_dict[model][data_preprocessing]['avg_pc_list'].append(float(avg_pc))
                    result_dict[model][data_preprocessing]['accuracy_list'].append(float(accuracy))
                    result_dict[model][data_preprocessing]['avg_f1_list'].append(float(avg_f1))
                else:
                    print ("Error! Please print the right mode")
                    sys.exit()
    # ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# (3.) box_plot
# ----------------------------------------------------------------------------------------------------------------------
#title = 'MLP {} performance under different number of trails'.format(classifier)
if mode =='clf':
    metrics_name_list = ['avg_f1_list', 'accuracy_list']
    ylim_range = (0,0.8)
    model_result_box_plot(result_dict, model_list, data_preprocessing_list, metrics_name_list, title ='',
                          x_label = '', ylim_range = ylim_range
    )
elif mode =='reg':
    metrics_name_list = ['rmse_list']
    ylim_range = (0.0, 1.3)
    xlim_range = (0,7)
    model_result_box_plot(result_dict, model_list, data_preprocessing_list, metrics_name_list, title ='',
                          x_label = '',ylim_range = ylim_range,xlim_range=xlim_range
    )
    metrics_name_list = ['avg_pc_list']
    ylim_range = (-0.01, 0.02)
    xlim_range = (0,7)
    model_result_box_plot(result_dict, model_list, data_preprocessing_list, metrics_name_list, title ='',
                          x_label = '',ylim_range = ylim_range,xlim_range=xlim_range
    )
# ----------------------------------------------------------------------------------------------------------------------