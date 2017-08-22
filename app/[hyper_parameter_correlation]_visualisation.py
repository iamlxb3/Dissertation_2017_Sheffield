# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Visualisation
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# (c) 2017 PJS, University of Sheffield, iamlxb3@gmail.com
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# ==========================================================================================================
# general package import
# ==========================================================================================================
import sys
import random
import os
import numpy as np
import re
import collections
import matplotlib.pyplot as plt
# ==========================================================================================================

# ==========================================================================================================
# ADD SYS PATH
# ==========================================================================================================
parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ==========================================================================================================

# ==========================================================================================================
# local package import
# ==========================================================================================================

# ==========================================================================================================

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# ----------------------------------------------------------------------------------------------------------
# (a.) get data set
# ----------------------------------------------------------------------------------------------------------
data_set = 'dow_jones_extended'
#classifier = 'adaboost_regressor'
model = 'classifier'  #regressor, bagging_regressor, bagging_classifier, adaboost_regressor
data_preprocessing = 'pca_standardization'
# data_preprocessing = 'origin'
# data_preprocessing = 'pca'
# data_preprocessing = 'standardization'

result_folder_temp = os.path.join(parent_folder, 'hyper_parameter_correlation_test', data_set, model, data_preprocessing)
# ----------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------
# (b.) save folder
# ----------------------------------------------------------------------------------------------------------
save_folder = os.path.join(parent_folder, 'hyper_parameter_correlation_visualisation', data_set)
# ----------------------------------------------------------------------------------------------------------


#
#[0] - experiment
#[1] - trail
#[2] - random_state_total
#[3] - pca_n_component
#[4] - activation_function
#[5] - alpha
#[6] - learning_rate
#[7] - learning_rate_init
#[8] - early_stopping
#[9] - validation_fraction
#[10] - hidden_layer_write_str
#[-1] - random_state
#[-2] - hidden_layer_nodes
#[-3] - hidden_layer_depth


hyper_parameter_dict = {'pca_n_component':3,
                        'activation_function':4,
                        'alpha':5,
                        'learning_rate':6,
                        'learning_rate_init_constant':7,
                        'learning_rate_init_invscaling': 7,
                        'early_stopping':8,
                        'validation_fraction':9,
                        'hidden_layer_write_str':10,
                        'random_state': -1,
                        'hidden_layer_depth': -3,
                        }

early_stopping_dict = {'True': 1, 'False': 0}
activation_function_dict = {'identity': 0, 'logistic': 1, 'tanh': 3, 'relu': 4}
learning_rate_dict = {'invscaling': 0, 'constant': 1}
# ==========================================================================================================
# (1.) [learning rate, pca_n_component, alpha] correlation
# ==========================================================================================================
hyper_parameter_list = ['learning_rate']
mode = 'avg' #best，avg

for hyper_parameter in hyper_parameter_list:

    # (1.) read the input folder
    result_folder = os.path.join(result_folder_temp, hyper_parameter)
    #

    # (2.) get file path list
    file_name_list = os.listdir(result_folder)
    file_path_list = [os.path.join(result_folder, x) for x in file_name_list]
    #

    # (3.) create hyper_parameter dict
    hyper_parameter_index = hyper_parameter_dict[hyper_parameter]
    hyper_parameter_dict = collections.defaultdict(lambda:collections.defaultdict(lambda:[]))
    #

    # (4.) read results into dict
    for i, file_path in enumerate(file_path_list):
        file_name = file_name_list[i]
        file_name = file_name[:-4]
        hyper_parameter_list_from_file_name = file_name.split('_')
        if hyper_parameter == 'early_stopping' or hyper_parameter == 'activation_function'\
            or hyper_parameter == 'learning_rate':
            hyper_parameter_value = hyper_parameter_list_from_file_name[hyper_parameter_index]


        else:
            hyper_parameter_value = float(hyper_parameter_list_from_file_name[hyper_parameter_index])
        with open (file_path,'r') as f:
            metrics_list = f.readlines()[0].strip().split(',')
            loss = float(metrics_list[0]) # result from the moving-window cross validation
            n_iter = float(metrics_list[1])
            f1 = float(metrics_list[2])
            accuracy = float(metrics_list[3])
            hyper_parameter_dict[hyper_parameter_value]['loss'].append(loss)
            hyper_parameter_dict[hyper_parameter_value]['n_iter'].append(n_iter)
            hyper_parameter_dict[hyper_parameter_value]['f1'].append(f1)
            hyper_parameter_dict[hyper_parameter_value]['accuracy'].append(accuracy)



    # ----------------------------------------------------------------------------------------------------------
    # plot
    # ----------------------------------------------------------------------------------------------------------


    # (0.) get the X, Y for plot
    hyper_parameter_value_list = []
    hyper_parameter_xticks = []
    f1_mean_list = []
    f1_std_list = []
    accuracy_mean_list = []
    accuracy_std_list = []



    for hyper_parameter_value in sorted(list(hyper_parameter_dict.keys())):
        if hyper_parameter == 'early_stopping':
            hyper_parameter_value_temp_id = early_stopping_dict[hyper_parameter_value]
            hyper_parameter_value_list.append(hyper_parameter_value_temp_id)
            hyper_parameter_xticks.append(hyper_parameter_value)
        elif hyper_parameter == 'activation_function':
            hyper_parameter_value_temp_id = activation_function_dict[hyper_parameter_value]
            hyper_parameter_value_list.append(hyper_parameter_value_temp_id)
            hyper_parameter_xticks.append(hyper_parameter_value)
        elif hyper_parameter == 'learning_rate':
            hyper_parameter_value_temp_id = learning_rate_dict[hyper_parameter_value]
            hyper_parameter_value_list.append(hyper_parameter_value_temp_id)
            hyper_parameter_xticks.append(hyper_parameter_value)
        else:
            hyper_parameter_value_list.append(hyper_parameter_value)

        f1_list = hyper_parameter_dict[hyper_parameter_value]['f1']

        f1_mean = np.average(f1_list)
        f1_std = np.std(f1_list)
        accuracy_list = hyper_parameter_dict[hyper_parameter_value]['accuracy']
        accuracy_mean = np.average(accuracy_list)
        accuracy_std = np.std(accuracy_list)

        f1_mean_list.append(f1_mean)
        f1_std_list.append(f1_std)
        accuracy_mean_list.append(accuracy_mean)
        accuracy_std_list.append(accuracy_std)


    #

    # ax1
    f1, (ax1,ax2) = plt.subplots(2, sharex=True, sharey=True)
    #ax1.plot(hyper_parameter_value_list, f1_mean_list, 'x', label = 'average_f1_mean')
    if hyper_parameter == 'early_stopping' or hyper_parameter == 'activation_function' \
            or hyper_parameter == 'learning_rate':
        plt.xticks(hyper_parameter_value_list, hyper_parameter_xticks)
    ax1.errorbar(hyper_parameter_value_list, f1_mean_list, f1_std_list, fmt='-o', capsize=5, label = 'average_f1')
    ax1.set_title('{} correlation with {}'.format(hyper_parameter, data_preprocessing))
    ax1.set_xlabel('{}'.format(hyper_parameter))
    ax1.legend()
    #

    # ax2
    #ax2.plot(hyper_parameter_value_list, accuracy_mean_list, 'x', label = 'accuracy_mean')
    ax2.errorbar(hyper_parameter_value_list, accuracy_mean_list, accuracy_std_list, fmt='-o', capsize=5, label = 'accuracy')
    ax2.set_xlabel('{}'.format(hyper_parameter))
    ax2.legend()
    #

    save_fig_path = '{}-{}-{}-{}.png'.format(data_preprocessing, model, hyper_parameter, mode)
    save_fig_path = os.path.join(save_folder, save_fig_path)
    plt.savefig(save_fig_path)
    # ----------------------------------------------------------------------------------------------------------

plt.show()
# ==========================================================================================================














