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
data_set = 'dow_jones'
#classifier = 'adaboost_regressor'
model = 'classifier'  #regressor, bagging_regressor, bagging_classifier, adaboost_regressor
data_preprocessing = 'pca_standardization'
# data_preprocessing = 'origin'
# data_preprocessing = 'pca'
# data_preprocessing = 'standardization'

result_folder = os.path.join(parent_folder, 'hyper_parameter_test', data_set, model, 'RESULT')
# ----------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------
# (b.) save folder
# ----------------------------------------------------------------------------------------------------------
save_folder = os.path.join(parent_folder, 'result_visualisation', data_set)
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
#
hyper_parameter_dict = {'pca_n_component':3,
                        'activation_function':4,
                        'alpha':5,
                        'learning_rate':6,
                        'learning_rate_init':7,
                        'early_stopping':8,
                        'validation_fraction':9,
                        'hidden_layer_write_str':10,
                        }


# ==========================================================================================================
# (1.) [learning rate, pca_n_component, alpha] correlation
# ==========================================================================================================
hyper_parameter_list = ['learning_rate_init', 'pca_n_component', 'alpha']
mode = 'avg' #best，avg

for hyper_parameter in hyper_parameter_list:
    # hyper parameters
    hyper_parameter_index = hyper_parameter_dict[hyper_parameter]
    hyper_parameter_chosen_list = []
    #

    sort_by_avg_file = os.path.join(result_folder, '{}_sort_by_average_accuracy.txt'.format(data_preprocessing))
    sort_by_best_file = os.path.join(result_folder, '{}_sort_by_best_accuracy.txt'.format(data_preprocessing))
    file = sort_by_avg_file

    if mode == 'avg':
        file = sort_by_avg_file
    elif mode == 'best':
        file = sort_by_best_file




    # result
    avg_f1_list = []
    accuracy_list = []
    loss_list = []
    iteration_step_list = []
    #

    # push data into list
    with open(file, 'r') as f:
        temp_hyper_parameter_list = []
        metric_result_list = []
        is_result = False
        for line in f:
            line = line.strip()
            if line == '===========================':
                #
                hyper_parameter_chosen_list.append(temp_hyper_parameter_list[hyper_parameter_index])
                avg_f1_list.append(metric_result_list[2])
                accuracy_list.append(metric_result_list[3])
                loss_list.append(metric_result_list[0])
                iteration_step_list.append(metric_result_list[1])
                #print ("metric_result_list: ", metric_result_list)
                #
                is_result = False
                temp_hyper_parameter_list = []
            elif line == '---------------------------':
                is_result = True
                metric_result_list = []
            else:
                if not is_result:
                    temp_hyper_parameter_list.append(line.strip())
                else:
                    #print('line: ', line)
                    value = re.findall(r': ([0-9\.]+)', line)[0]
                    if mode == 'avg':
                        var = re.findall(r': ([0-9\.]+)', line)[1]
                        std = re.findall(r': ([0-9\.]+)', line)[2]
                    if mode == 'best':
                        metric_result_list.append(value)
                    elif mode == 'avg':
                        metric_result_list.append((value,var,std))


    # ----------------------------------------------------------------------------------------------------------
    # plot
    # ----------------------------------------------------------------------------------------------------------

    f1, (ax1,ax2) = plt.subplots(2, sharex=True, sharey=True)


    if mode == 'avg':
        avg_f1_list = [x[0] for x in avg_f1_list]
        accuracy_list = [x[0] for x in accuracy_list]

    # ax1
    ax1.plot(hyper_parameter_chosen_list, avg_f1_list, 'x', label = 'average_f1')
    ax1.set_title('{} correlation'.format(hyper_parameter))
    ax1.set_xlabel('{}'.format(hyper_parameter))
    ax1.legend()
    #

    # ax2
    ax2.plot(hyper_parameter_chosen_list, accuracy_list, 'x', label = 'accuracy')
    #ax2.set_title('{} correlation'.format(hyper_parameter))
    ax2.set_xlabel('{}'.format(hyper_parameter))
    ax2.legend()
    #

    save_fig_path = '{}-{}-{}-{}.png'.format(data_preprocessing, model, hyper_parameter, mode)
    save_fig_path = os.path.join(save_folder, save_fig_path)
    plt.savefig(save_fig_path)
    # ----------------------------------------------------------------------------------------------------------
plt.show()
# ==========================================================================================================


# ==========================================================================================================
# (2.) # early_stopping, validation_fraction
# ==========================================================================================================
hyper_parameter_list = ['early_stopping', 'validation_fraction']
mode = 'avg' #best，avg

for hyper_parameter in hyper_parameter_list:
    # hyper parameters
    hyper_parameter_index = hyper_parameter_dict[hyper_parameter]
    hyper_parameter_chosen_list = []
    #

    sort_by_avg_file = os.path.join(result_folder, '{}_sort_by_average_accuracy.txt'.format(data_preprocessing))
    sort_by_best_file = os.path.join(result_folder, '{}_sort_by_best_accuracy.txt'.format(data_preprocessing))
    file = sort_by_avg_file

    if mode == 'avg':
        file = sort_by_avg_file
    elif mode == 'best':
        file = sort_by_best_file




    # result
    avg_f1_list = []
    accuracy_list = []
    loss_list = []
    iteration_step_list = []
    #

    # push data into list
    with open(file, 'r') as f:
        temp_hyper_parameter_list = []
        metric_result_list = []
        is_result = False
        for line in f:
            line = line.strip()
            if line == '===========================':
                #
                hyper_parameter_chosen_list.append(temp_hyper_parameter_list[hyper_parameter_index])
                avg_f1_list.append(metric_result_list[2])
                accuracy_list.append(metric_result_list[3])
                loss_list.append(metric_result_list[0])
                iteration_step_list.append(metric_result_list[1])
                #print ("metric_result_list: ", metric_result_list)
                #
                is_result = False
                temp_hyper_parameter_list = []
            elif line == '---------------------------':
                is_result = True
                metric_result_list = []
            else:
                if not is_result:
                    temp_hyper_parameter_list.append(line.strip())
                else:
                    #print('line: ', line)
                    value = re.findall(r': ([0-9\.]+)', line)[0]
                    if mode == 'avg':
                        var = re.findall(r': ([0-9\.]+)', line)[1]
                        std = re.findall(r': ([0-9\.]+)', line)[2]
                    if mode == 'best':
                        metric_result_list.append(value)
                    elif mode == 'avg':
                        metric_result_list.append((value,var,std))


    # ----------------------------------------------------------------------------------------------------------
    # plot
    # ----------------------------------------------------------------------------------------------------------

    f1, (ax1,ax2) = plt.subplots(2, sharex=True, sharey=True)


    if mode == 'avg':
        avg_f1_list = [x[0] for x in avg_f1_list]
        accuracy_list = [x[0] for x in accuracy_list]

    if hyper_parameter == 'early_stopping': #, validation_fraction
        hyper_parameter_chosen_list = [0 if x == 'False' else 1 for x in hyper_parameter_chosen_list]
    elif hyper_parameter == 'validation_fraction':
        hyper_parameter_chosen_list = [float(x) for x in hyper_parameter_chosen_list]
        temp_hyper_parameter_chosen_list = []
        temp_avg_f1_list = []
        temp_accuracy_list = []
        for i, x in enumerate(hyper_parameter_chosen_list):
            if x > 0:
                temp_hyper_parameter_chosen_list.append(x)
                temp_avg_f1_list.append(avg_f1_list[i])
                temp_accuracy_list.append(accuracy_list[i])
        hyper_parameter_chosen_list = temp_hyper_parameter_chosen_list
        avg_f1_list = temp_avg_f1_list
        accuracy_list = temp_accuracy_list

    # ax1
    ax1.plot(hyper_parameter_chosen_list, avg_f1_list, 'x', label = 'average_f1')
    ax1.set_title('{} correlation'.format(hyper_parameter))
    ax1.set_xlabel('{}'.format(hyper_parameter))
    ax1.legend()
    #

    # ax2
    ax2.plot(hyper_parameter_chosen_list, accuracy_list, 'x', label = 'accuracy')
    #ax2.set_title('{} correlation'.format(hyper_parameter))
    ax2.set_xlabel('{}'.format(hyper_parameter))
    ax2.legend()
    #

    save_fig_path = '{}-{}-{}-{}.png'.format(data_preprocessing, model, hyper_parameter, mode)
    save_fig_path = os.path.join(save_folder, save_fig_path)
    plt.savefig(save_fig_path)
    # ----------------------------------------------------------------------------------------------------------

plt.show()
# ==========================================================================================================


# ==========================================================================================================
# (3.) # topology
# ==========================================================================================================
hyper_parameter_list = ['depth_of_hidden_layer','total_nodes_in_hidden_layer']
mode = 'avg' #best，avg

for hyper_parameter in hyper_parameter_list:
    # hyper parameters
    hyper_parameter_index = hyper_parameter_dict['hidden_layer_write_str']
    hyper_parameter_chosen_list = []
    #

    sort_by_avg_file = os.path.join(result_folder, '{}_sort_by_average_accuracy.txt'.format(data_preprocessing))
    sort_by_best_file = os.path.join(result_folder, '{}_sort_by_best_accuracy.txt'.format(data_preprocessing))
    file = sort_by_avg_file

    if mode == 'avg':
        file = sort_by_avg_file
    elif mode == 'best':
        file = sort_by_best_file




    # result
    avg_f1_list = []
    accuracy_list = []
    loss_list = []
    iteration_step_list = []
    #

    # push data into list
    with open(file, 'r') as f:
        temp_hyper_parameter_list = []
        metric_result_list = []
        is_result = False
        for line in f:
            line = line.strip()
            if line == '===========================':
                #
                if hyper_parameter == 'depth_of_hidden_layer':
                    hyper_parameter_value = len(temp_hyper_parameter_list) - hyper_parameter_index
                elif hyper_parameter == 'total_nodes_in_hidden_layer':
                    hyper_parameter_value = temp_hyper_parameter_list[hyper_parameter_index:]
                    hyper_parameter_value = sum([float(x) for x in hyper_parameter_value])

                hyper_parameter_chosen_list.append(hyper_parameter_value)
                avg_f1_list.append(metric_result_list[2])
                accuracy_list.append(metric_result_list[3])
                loss_list.append(metric_result_list[0])
                iteration_step_list.append(metric_result_list[1])
                #print ("metric_result_list: ", metric_result_list)
                #
                is_result = False
                temp_hyper_parameter_list = []
            elif line == '---------------------------':
                is_result = True
                metric_result_list = []
            else:
                if not is_result:
                    temp_hyper_parameter_list.append(line.strip())
                else:
                    #print('line: ', line)
                    value = re.findall(r': ([0-9\.]+)', line)[0]
                    if mode == 'avg':
                        var = re.findall(r': ([0-9\.]+)', line)[1]
                        std = re.findall(r': ([0-9\.]+)', line)[2]
                    if mode == 'best':
                        metric_result_list.append(value)
                    elif mode == 'avg':
                        metric_result_list.append((value,var,std))


    # ----------------------------------------------------------------------------------------------------------
    # plot
    # ----------------------------------------------------------------------------------------------------------

    f1, (ax1,ax2) = plt.subplots(2, sharex=True, sharey=True)


    if mode == 'avg':
        avg_f1_list = [x[0] for x in avg_f1_list]
        accuracy_list = [x[0] for x in accuracy_list]

    if hyper_parameter == 'early_stopping': #, validation_fraction
        hyper_parameter_chosen_list = [0 if x == 'False' else 1 for x in hyper_parameter_chosen_list]
    elif hyper_parameter == 'validation_fraction':
        hyper_parameter_chosen_list = [float(x) for x in hyper_parameter_chosen_list]
        temp_hyper_parameter_chosen_list = []
        temp_avg_f1_list = []
        temp_accuracy_list = []
        for i, x in enumerate(hyper_parameter_chosen_list):
            if x > 0:
                temp_hyper_parameter_chosen_list.append(x)
                temp_avg_f1_list.append(avg_f1_list[i])
                temp_accuracy_list.append(accuracy_list[i])
        hyper_parameter_chosen_list = temp_hyper_parameter_chosen_list
        avg_f1_list = temp_avg_f1_list
        accuracy_list = temp_accuracy_list

    # ax1
    ax1.plot(hyper_parameter_chosen_list, avg_f1_list, 'x', label = 'average_f1')
    ax1.set_title('{} correlation'.format(hyper_parameter))
    ax1.set_xlabel('{}'.format(hyper_parameter))
    ax1.legend()
    #

    # ax2
    ax2.plot(hyper_parameter_chosen_list, accuracy_list, 'x', label = 'accuracy')
    #ax2.set_title('{} correlation'.format(hyper_parameter))
    ax2.set_xlabel('{}'.format(hyper_parameter))
    ax2.legend()
    #

    save_fig_path = '{}-{}-{}-{}.png'.format(data_preprocessing, model, hyper_parameter, mode)
    save_fig_path = os.path.join(save_folder, save_fig_path)
    plt.savefig(save_fig_path)
    # ----------------------------------------------------------------------------------------------------------

plt.show()
# ==========================================================================================================















