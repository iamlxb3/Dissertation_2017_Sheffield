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
data_set = 'dow_jones_index_extended'
mode = 'clf'
classification_list = ['classifier','bagging_classifier','regressor','bagging_regressor','adaboost_regressor',
                       'random_forest_classifier']
regression_list = ['regressor','bagging_regressor','adaboost_regressor']

classifier_list = ['classifier','bagging_classifier','random_forest_classifier']
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
                feature_str = file_name[0:-4].strip()
                feature_list = feature_str.split('_')
                if model in classifier_list:
                    loss = line_list[0]
                    avg_iter_num = line_list[1]
                    avg_f1 = line_list[2]
                    accuracy = line_list[3]
                    result_dict[model][data_preprocessing]['loss_list'].append(float(loss))
                    result_dict[model][data_preprocessing]['avg_iter_num_list'].append(float(avg_iter_num))
                    result_dict[model][data_preprocessing]['avg_f1_list'].append(float(avg_f1))
                    result_dict[model][data_preprocessing]['accuracy_list'].append(float(accuracy))
                    result_dict[model][data_preprocessing]['feature_list'].append(feature_list)

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
                    result_dict[model][data_preprocessing]['accuracy_list'].append(float(accuracy))
                    result_dict[model][data_preprocessing]['feature_list'].append(feature_list)

                else:
                    print ("Error! Please print the right mode")
                    sys.exit()
    # ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# (4.) save the results for each classifier
# ----------------------------------------------------------------------------------------------------------------------
if mode == 'clf':
    sort_by = ['avg_f1_list', 'accuracy_list']
elif mode == 'reg':
    sort_by = ['rmse_list', 'avg_pc_list']
else:
    print ("Please enter the right mode!!")
    sys.exit()

for model, data_preprocessing_dict in result_dict.items():
    for metric in sort_by:
        save_name = '{}_validation_result_[{}].csv'.format(model, metric)
        save_path = os.path.join(parent_folder, 'results', 'model_results', save_name)
        metric_value_list = []
        feature_value_list = []
        for data_preprocessing in data_preprocessing_list:
            metric_value_list.extend(data_preprocessing_dict[data_preprocessing][metric])
            feature_value_list.extend(data_preprocessing_dict[data_preprocessing]['feature_list'])
        zip_list = list(zip(metric_value_list, feature_value_list))
        if metric == 'rmse_list':
            sorted_zip_list = sorted(zip_list, key = lambda x:x[0])
        else:
            sorted_zip_list = sorted(zip_list, key = lambda x:x[0], reverse=True)

        # save file
        with open (save_path, 'w') as f:
            for metric, feature_list in sorted_zip_list:
                f.write(','.join(feature_list))
                f.write('\n')
                f.write(str(metric))
                f.write('\n')
        print ("Save results of {} successfully!".format(save_name))
# ----------------------------------------------------------------------------------------------------------------------