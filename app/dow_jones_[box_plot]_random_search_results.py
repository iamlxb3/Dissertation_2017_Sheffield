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
from stock_box_plot2 import stock_metrics_result_box_plot
# ==========================================================================================================

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# ----------------------------------------------------------------------------------------------------------------------
# read data into dictionary
# ----------------------------------------------------------------------------------------------------------------------
data_set = 'dow_jones_index_extended'
mode = 'clf'
#classifier = 'classifier'
classifier = 'classifier'
data_preprocessing = 'pca_standardization'
# data_preprocessing = 'origin'
# data_preprocessing = 'pca'
# data_preprocessing = 'standardization'

# input path
hyper_parameter_folder = os.path.join(parent_folder, 'hyper_parameter_test',data_set, classifier, data_preprocessing)
file_name_list = os.listdir(hyper_parameter_folder)
file_path_list = [os.path.join(hyper_parameter_folder, x) for x in file_name_list]
#

# save path


#


# ----------------------------------------------------------------------------------------------------------------------
# (1.) push every model's result to dict
# ----------------------------------------------------------------------------------------------------------------------
result_dict = collections.defaultdict(lambda :{})
for i, file in enumerate(file_path_list):
    with open (file, 'r') as f:
        file_name = file_name_list[i]
        if file_name == '.gitignore' or  file_name == 'feature_selection.txt':
            continue
        for j, line in enumerate(f):
            unique_id = "{}_{}".format(i,j)
            line_list = line.strip().split(',')
            if mode == 'clf':
                loss = line_list[0]
                avg_iter_num = line_list[1]
                avg_f1 = line_list[2]
                accuracy = line_list[3]
                result_dict[unique_id]['loss'] = float(loss)
                result_dict[unique_id]['avg_iter_num'] = float(avg_iter_num)
                result_dict[unique_id]['avg_f1'] = float(avg_f1)
                result_dict[unique_id]['accuracy'] = float(accuracy)
            elif mode == 'reg':
                loss = line_list[0]
                avg_iter_num = line_list[1]
                rmse = line_list[2]
                avg_pc = line_list[3]
                accuracy = line_list[4]
                avg_f1 = line_list[5]
                result_dict[unique_id]['loss'] = float(loss)
                result_dict[unique_id]['avg_iter_num'] = float(avg_iter_num)
                result_dict[unique_id]['rmse'] = float(rmse)
                result_dict[unique_id]['avg_pc'] = float(avg_pc)
                result_dict[unique_id]['accuracy'] = float(accuracy)
                result_dict[unique_id]['avg_f1'] = float(avg_f1)
            else:
                print ("Error! Please print the right mode")
                sys.exit()
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# (2.) get the data for plot for each experiment
# ----------------------------------------------------------------------------------------------------------------------
NUMBER_OF_TRAILS_LIST = [1,2,4,8,16,32,64,128]
metrics_result_dict = collections.defaultdict(lambda :collections.defaultdict(lambda :[]))
for trails_num in NUMBER_OF_TRAILS_LIST:
    unique_id_list = sorted(list(result_dict.keys()))
    if len(unique_id_list)%trails_num == 0:
        experiment_num = int(len(unique_id_list)/trails_num)
    else:
        print ("experiment_num: {} is not int! unique_id_len: {}".format(len(unique_id_list)/trails_num, len(unique_id_list)))
        sys.exit()
    for i in range(0,experiment_num):
        random.seed(1)
        unique_id_chosen = random.sample(unique_id_list, trails_num)

        # delete the chosen_id
        for unique_id in unique_id_chosen:
            unique_id_list.remove(unique_id)
        #

        avg_f1_list = []
        accuracy_list = []
        avg_pc_list = []
        rmse_list = []


        for unique_id in unique_id_chosen:
            if mode == 'clf':
                loss = result_dict[unique_id]['loss']
                avg_iter_num = result_dict[unique_id]['avg_iter_num']
                avg_f1 = result_dict[unique_id]['avg_f1']
                accuracy = result_dict[unique_id]['accuracy']
                #
                avg_f1_list.append(avg_f1)
                accuracy_list.append(accuracy)

            elif mode == 'reg':
                loss = result_dict[unique_id]['loss']
                avg_iter_num = result_dict[unique_id]['avg_iter_num']
                rmse = result_dict[unique_id]['rmse']
                avg_pc = result_dict[unique_id]['avg_pc']
                accuracy = result_dict[unique_id]['accuracy']
                avg_f1 = result_dict[unique_id]['avg_f1']
                #
                avg_f1_list.append(avg_f1)
                accuracy_list.append(accuracy)
                avg_pc_list.append(avg_pc)
                rmse_list.append(rmse)
            else:
                print ("Error")

        if mode == 'clf':
            best_avg_f1 = max(avg_f1_list)
            best_accuracy= max(accuracy_list)
            metrics_result_dict[trails_num]['avg_f1'].append(best_avg_f1)
            metrics_result_dict[trails_num]['accuracy'].append(best_accuracy)

        elif mode == 'reg':
            best_rmse= max(rmse_list)
            best_avg_pc = max(avg_pc_list)
            best_avg_f1 = max(avg_f1_list)
            best_accuracy= max(accuracy_list)
            metrics_result_dict[trails_num]['avg_f1'].append(best_avg_f1)
            metrics_result_dict[trails_num]['accuracy'].append(best_accuracy)
            metrics_result_dict[trails_num]['rmse'].append(best_rmse)
            metrics_result_dict[trails_num]['avg_pc'].append(best_avg_pc)

print ("Mode: '{}', data_preprocessing: '{}', All data for '{}' of '{}' ready for plot!".format(mode, data_preprocessing,
                                                                               data_set, classifier))
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# (3.) box_plot
# ----------------------------------------------------------------------------------------------------------------------
title = 'MLP {} performance under different number of trails'.format(classifier)
stock_metrics_result_box_plot(metrics_result_dict, NUMBER_OF_TRAILS_LIST, ['avg_f1', 'accuracy'], title =title
)


# ----------------------------------------------------------------------------------------------------------------------