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
# clf_path = os.path.join(parent_folder, 'classifiers', 'mlp')
# sys.path.append(clf_path)
# ==========================================================================================================

# ==========================================================================================================
# local package import
# ==========================================================================================================

# ==========================================================================================================

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# # ----------------------------------------------------------------------------------------------------------------------
# # rename files
# # ----------------------------------------------------------------------------------------------------------------------
# hyper_parameter_folder = os.path.join(parent_folder, 'hyper_parameter_test')
#
# file_name_list = os.listdir(hyper_parameter_folder)
#
#
# for old_file_name in file_name_list:
#     file_name = old_file_name[10:]
#     file_path = os.path.join(hyper_parameter_folder, file_name)
#     old_file_path = os.path.join(hyper_parameter_folder, old_file_name)
#     os.rename(old_file_path, file_path)
# # ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# read data into dictionary
# ----------------------------------------------------------------------------------------------------------------------
data_set = 'dow_jones'
classifier = 'adaboost_regressor'
data_preprocessing = 'pca_standardization'
# data_preprocessing = 'origin'
# data_preprocessing = 'pca'
# data_preprocessing = 'standardization'


hyper_parameter_folder = os.path.join(parent_folder, 'hyper_parameter_test',data_set, classifier, data_preprocessing)

file_name_list = os.listdir(hyper_parameter_folder)
hyper_parameter_avg_dict = collections.defaultdict(lambda :{})
hyper_parameter_best_list = []

for file_name in file_name_list:
    if file_name == '.gitignore':
        continue
    file_path = os.path.join(hyper_parameter_folder, file_name)
    file_name = file_name[0:-4]
    hyper_parameter_tuple = tuple(file_name.split('_'))
    with open (file_path, 'r') as f:
        f_readlines_list = f.readlines()
        f_readlines_list = [line.strip().split(',') for line in f_readlines_list]

        regressor_eva_metrics_list = list(zip(*f_readlines_list))
        #regressor_eva_metrics_list = regressor_eva_metrics_list[0:-1]  #get rid of the random state



        # convert tuple to list
        regressor_eva_metrics_list = [list(x) for x in regressor_eva_metrics_list]

        # convert str to float
        for x_list in regressor_eva_metrics_list:
            for i, x in enumerate(x_list):
                x_list[i] = float(x)
        #

        regressor_eva_metrics_avg_value_list = [np.average(x) for x in regressor_eva_metrics_list]
        regressor_eva_metrics_avg_var_list = [np.var(x) for x in regressor_eva_metrics_list]
        regressor_eva_metrics_avg_std_list = [np.std(x) for x in regressor_eva_metrics_list]
        regressor_eva_metrics_avg_list = list(zip(regressor_eva_metrics_avg_value_list, regressor_eva_metrics_avg_var_list,
                                             regressor_eva_metrics_avg_std_list))


    hyper_parameter_avg_dict[hyper_parameter_tuple] = regressor_eva_metrics_avg_list

    # best
    for eva_result in f_readlines_list:
        eva_result = [float(x) for x in eva_result]
        hyper_parameter_best_list.append((hyper_parameter_tuple, eva_result))

    # best
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# get the best hyper-parameters based on the average performance of different weight initialization
# ----------------------------------------------------------------------------------------------------------------------
# sort by average
hyper_parameter_result_save_folder = os.path.join(parent_folder, 'hyper_parameter_test',data_set, classifier, 'RESULT')
#
hyper_parameter_avg_loss_save_path = "{}_sort_by_average_loss.txt".format(data_preprocessing)
hyper_parameter_avg_loss_save_path = os.path.join(hyper_parameter_result_save_folder, hyper_parameter_avg_loss_save_path)
hyper_parameter_avg_rmse_save_path = "{}_sort_by_average_rmse.txt".format(data_preprocessing)
hyper_parameter_avg_rmse_save_path = os.path.join(hyper_parameter_result_save_folder, hyper_parameter_avg_rmse_save_path)
hyper_parameter_avg_avg_pc_save_path = "{}_sort_by_average_avg_pc.txt".format(data_preprocessing)
hyper_parameter_avg_avg_pc_save_path = os.path.join(hyper_parameter_result_save_folder, hyper_parameter_avg_avg_pc_save_path)
hyper_parameter_avg_hit_save_path = "{}_sort_by_average_hit.txt".format(data_preprocessing)
hyper_parameter_avg_hit_save_path = os.path.join(hyper_parameter_result_save_folder, hyper_parameter_avg_hit_save_path)
#
# sort by best
hyper_parameter_lowest_loss_save_path = "{}_sort_by_lowest_loss.txt".format(data_preprocessing)
hyper_parameter_lowest_loss_save_path = os.path.join(hyper_parameter_result_save_folder, hyper_parameter_lowest_loss_save_path)
hyper_parameter_best_rmse_save_path = "{}_sort_by_best_rmse.txt".format(data_preprocessing)
hyper_parameter_best_rmse_save_path = os.path.join(hyper_parameter_result_save_folder, hyper_parameter_best_rmse_save_path)
hyper_parameter_best_avg_pc_save_path = "{}_sort_by_best_avg_pc.txt".format(data_preprocessing)
hyper_parameter_best_avg_pc_save_path = os.path.join(hyper_parameter_result_save_folder, hyper_parameter_best_avg_pc_save_path)
hyper_parameter_best_hit_save_path = "{}_sort_by_best_hit.txt".format(data_preprocessing)
hyper_parameter_best_hit_save_path = os.path.join(hyper_parameter_result_save_folder, hyper_parameter_best_hit_save_path)
#




# ----------------------------------------------------------------------------------------------------------------------
# save the hyper parameter result
def save_hyper_parameter_result_to_file_regressor(hyper_parameter_list, save_path, is_avg = True):
    with open(save_path, 'w') as f:
        for hyper_parameter_tuple, evaluation_metrics_tuple in hyper_parameter_list:
            if is_avg:
                f.write('===========================\n')
                hyper_parameter_tuple_str = '\n'.join(hyper_parameter_tuple)
                f.write(hyper_parameter_tuple_str + '\n')
                f.write('---------------------------\n')
                evaluation_metrics_list = ['' for x in list(evaluation_metrics_tuple)]
                #evaluation_metrics_list = [str(x) for x in list(evaluation_metrics_tuple)]
                evaluation_metrics_list[0] = 'loss: {}, var: {}, std: {}'.format(evaluation_metrics_tuple[0][0],evaluation_metrics_tuple[0][1],evaluation_metrics_tuple[0][2])
                evaluation_metrics_list[1] = 'iteration_step: {}, var: {}, std: {}'.format(evaluation_metrics_tuple[1][0],evaluation_metrics_tuple[1][1],evaluation_metrics_tuple[1][2])
                evaluation_metrics_list[2] = 'rmse: {}, var: {}, std: {}'.format(evaluation_metrics_tuple[2][0],evaluation_metrics_tuple[2][1],evaluation_metrics_tuple[2][2])
                evaluation_metrics_list[3] = 'avg_pc: {}, var: {}, std: {}'.format(evaluation_metrics_tuple[3][0],evaluation_metrics_tuple[3][1],evaluation_metrics_tuple[3][2])
                evaluation_metrics_list[4] = 'hit: {}, var: {}, std: {}'.format(evaluation_metrics_tuple[4][0],evaluation_metrics_tuple[4][1],evaluation_metrics_tuple[4][2])

                evaluation_metrics_tuple_str = '\n'.join(evaluation_metrics_list)
                f.write(evaluation_metrics_tuple_str + '\n')
            else:
                f.write('===========================\n')
                hyper_parameter_tuple_str = '\n'.join(hyper_parameter_tuple)
                f.write(hyper_parameter_tuple_str + '\n')
                f.write('---------------------------\n')
                evaluation_metrics_list = [str(x) for x in list(evaluation_metrics_tuple)]
                evaluation_metrics_list[0] = 'loss: ' + evaluation_metrics_list[0]
                evaluation_metrics_list[1] = 'iteration_step: ' + evaluation_metrics_list[1]
                evaluation_metrics_list[2] = 'rmse: ' + evaluation_metrics_list[2]
                evaluation_metrics_list[3] = 'avg_pc: ' + evaluation_metrics_list[3]
                evaluation_metrics_list[4] = 'hit: ' + evaluation_metrics_list[4]
                evaluation_metrics_list[5] = 'random_state: ' + evaluation_metrics_list[5]
                evaluation_metrics_tuple_str = '\n'.join(evaluation_metrics_list)
                f.write(evaluation_metrics_tuple_str + '\n')
# ----------------------------------------------------------------------------------------------------------------------



# ----------------------------------------------------------------------------------------------------------------------
# save the hyper_parameter by average value
# ----------------------------------------------------------------------------------------------------------------------
hyper_parameter_avg_list = list(hyper_parameter_avg_dict.items())

# average
# (0.) sort by loss
hyper_parameter_avg_list_sort_by_rmse = sorted(hyper_parameter_avg_list, key = lambda x:x[1][0])
save_hyper_parameter_result_to_file_regressor(hyper_parameter_avg_list_sort_by_rmse, hyper_parameter_avg_loss_save_path)

# (1.) sort by rmse
hyper_parameter_avg_list_sort_by_rmse = sorted(hyper_parameter_avg_list, key = lambda x:x[1][2])
save_hyper_parameter_result_to_file_regressor(hyper_parameter_avg_list_sort_by_rmse, hyper_parameter_avg_rmse_save_path)

# (2.) sort by avg_pc
hyper_parameter_avg_list_sort_by_avg_pc = sorted(hyper_parameter_avg_list, key = lambda x:x[1][3], reverse=True)
save_hyper_parameter_result_to_file_regressor(hyper_parameter_avg_list_sort_by_avg_pc, hyper_parameter_avg_avg_pc_save_path)

# (3.) sort by hit
hyper_parameter_avg_list_sort_by_hit = sorted(hyper_parameter_avg_list, key = lambda x:x[1][4], reverse=True)
save_hyper_parameter_result_to_file_regressor(hyper_parameter_avg_list_sort_by_hit, hyper_parameter_avg_hit_save_path)
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# save the hyper_parameter by best value
# ----------------------------------------------------------------------------------------------------------------------

# best
# (0.) sort by loss
hyper_parameter_best_list_sort_by_loss = sorted(hyper_parameter_best_list, key = lambda x:x[1][0])
save_hyper_parameter_result_to_file_regressor(hyper_parameter_best_list_sort_by_loss,
                                              hyper_parameter_lowest_loss_save_path, is_avg = False)

# (1.) sort by rmse
hyper_parameter_best_list_sort_by_rmse = sorted(hyper_parameter_best_list, key = lambda x:x[1][2])
save_hyper_parameter_result_to_file_regressor(hyper_parameter_best_list_sort_by_rmse,
                                              hyper_parameter_best_rmse_save_path, is_avg = False)

# (2.) sort by avg_pc
hyper_parameter_best_list_sort_by_avg_pc = sorted(hyper_parameter_best_list, key = lambda x:x[1][3], reverse=True)
save_hyper_parameter_result_to_file_regressor(hyper_parameter_best_list_sort_by_avg_pc,
                                              hyper_parameter_best_avg_pc_save_path, is_avg = False)

# (3.) sort by hit
hyper_parameter_best_list_sort_by_hit = sorted(hyper_parameter_best_list, key = lambda x:x[1][4], reverse=True)
save_hyper_parameter_result_to_file_regressor(hyper_parameter_best_list_sort_by_hit,
                                              hyper_parameter_best_hit_save_path, is_avg = False)
# ----------------------------------------------------------------------------------------------------------------------







