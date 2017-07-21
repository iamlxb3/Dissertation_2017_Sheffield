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
from trade_general_funcs import get_full_feature_switch_tuple
from trade_general_funcs import read_pca_component

# ==========================================================================================================

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



# ==========================================================================================================
# Build MLP classifier for a-share data, save the mlp to local
# ==========================================================================================================
print ("Build MLP regressor for dow-jones data!")
# (1.) build classifer
mlp_regressor1 = MlpTradeRegressor()
clsfy_name = 'dow_jone_hyper_parameter_MLP_regressor'
clf_path = os.path.join(parent_folder, 'trained_classifiers', clsfy_name)


# (2.) GET TRANINING SET
data_per = 1.0 # the percentage of data using for training and testing
dev_per = 0.0 # the percentage of data using for developing


input_folder = os.path.join('dow_jones_index','dow_jones_index_regression')
input_folder = os.path.join(parent_folder, 'data', input_folder)
feature_switch_tuple_all_1 = get_full_feature_switch_tuple(input_folder)

# (3.)
samples_feature_list, samples_value_list, \
date_str_list, stock_id_list = mlp_regressor1._feed_data(input_folder, data_per=data_per,
                                               feature_switch_tuple=feature_switch_tuple_all_1,
                                               is_random=False)


# (4.)
is_standardisation = True
is_PCA = True
shift_num = 5
shifting_size_percent = 0.1
pca_n_component = read_pca_component(input_folder)
include_top_list = [1]

mlp_regressor1.create_train_dev_vdict_window_shift(samples_feature_list, samples_value_list,
                                         date_str_list, stock_id_list, is_cv=True,
                                         shifting_size_percent=shifting_size_percent,
                                         shift_num=shift_num,
                                         is_standardisation=is_standardisation, is_PCA=is_PCA,
                                         pca_n_component=pca_n_component)




# (4.) HIDDEN LAYERS

hidden_layer_depth = [x for x in range(1, 3)]
hidden_layer_node = [x for x in range(100, 600)][::100]

hidden_layer_sizes_list = []
for layer_depth in hidden_layer_depth:
    hidden_layer_sizes_list_temp = list(itertools.product(hidden_layer_node, repeat=layer_depth))
    hidden_layer_sizes_list.extend(hidden_layer_sizes_list_temp)

random.shuffle(hidden_layer_sizes_list)
print ("hidden_layer_sizes_list: ", hidden_layer_sizes_list)
print ("Total {} hidden layers to test".format(len(hidden_layer_sizes_list)))

# (5.) learning_rate_init
learning_rate_init_list = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
learning_rate_init_list = learning_rate_init_list[::-1]
print ("learning_rate_init_list: ", learning_rate_init_list)

# (6.) learning_rate
learning_rate_list = ['constant', 'invscaling']
print ("learning_rate_list: ", learning_rate_list)

# (7.) learning_rate
early_stopping_list = [True, False]
print ("early_stopping_list: ", early_stopping_list)


hyper_parameter_list = list(itertools.product(hidden_layer_sizes_list, learning_rate_init_list, learning_rate_list,
                                              early_stopping_list))
hyper_parameter_size = len(hyper_parameter_list)
print ("hyper_parameter_size: ", hyper_parameter_size)

for i, (hidden_layer_sizes, learning_rate_init, learning_rate, early_stopping) in enumerate(hyper_parameter_list):
    tol = 1e-10
    n_iter_list = []
    loss_list = []
    rmse_list = []
    avg_pc_list = []
    random_state_list = []
    polar_percent_list = []
    random_state_total = 100

    for random_state in range(random_state_total):

        random_seed = 'window_shift'

        shift_n_iter_list = []
        shift_loss_list = []
        shift_rmse_list = []
        shift_avg_pc_list = []
        shift_polar_percent_list = []

        mlp_regressor1.set_regressor_test(hidden_layer_sizes, tol=tol, learning_rate_init=learning_rate_init,
                                          random_state=random_state,
                           verbose = False, learning_rate = learning_rate, early_stopping =early_stopping)
        for shift in mlp_regressor1.validation_dict[random_seed].keys():
            mlp_regressor1.trade_rs_cv_load_train_dev_data(random_seed, shift)
            mlp_regressor1.regressor_train(save_clsfy_path=clf_path)
            mrse, avg_price_change_tuple, polar_percent = mlp_regressor1.regressor_dev(save_clsfy_path=clf_path, is_cv=True, include_top_list=include_top_list)
            shift_n_iter_list.append(mlp_regressor1.mlp_regressor.n_iter_)
            shift_loss_list.append(mlp_regressor1.mlp_regressor.loss_)
            shift_rmse_list.append(mrse)
            shift_avg_pc_list.append(avg_price_change_tuple[0])
            shift_polar_percent_list.append(polar_percent)


        avg_n_iter = np.average(shift_n_iter_list)
        avg_loss= np.average(shift_loss_list)
        avg_rmse = np.average(np.average(shift_rmse_list))
        avg_pc = np.average(shift_avg_pc_list)
        avg_polar_percent = np.average(np.average(shift_polar_percent_list))



        random_state_list.append(random_state)
        n_iter_list.append(avg_n_iter)
        loss_list.append(avg_loss)
        rmse_list.append(avg_rmse)
        avg_pc_list.append(avg_pc)
        polar_percent_list.append(avg_polar_percent)

        print ("-----------------------------------------------------------------------------------")
        print ("random_state: {}|{}".format(random_state, random_state_total))
        print ("early_stopping: ", early_stopping)
        print ("learning_rate: ", learning_rate)
        print ("learning_rate_init: ", learning_rate_init)
        print ("hidden_layer_sizes: ", hidden_layer_sizes)
        print ("avg_n_iter: ", avg_n_iter)
        print ("avg_loss: ", avg_loss)
        print ("avg_rmse: ", avg_rmse)
        print ("avg_pc: ", avg_pc)
        print ("avg_polar_percent: ", avg_polar_percent)
        print ("Testing percent: {:.7f}%".format(100*i/hyper_parameter_size))

    # ==========================================================================================================
    write_list = list(zip(loss_list, n_iter_list, rmse_list, avg_pc_list, polar_percent_list, random_state_list))
    write_list = sorted(write_list, key = lambda x:x[0])


    hidden_layer_write_str = '_'.join([str(x) for x in hidden_layer_sizes])

    save_folder = os.path.join(parent_folder, 'loss_test')
    csv_file_path = os.path.join(save_folder, 'loss_test_{}_{}_{}_ES_{}.csv'.format(learning_rate, learning_rate_init,
                                                                              hidden_layer_write_str, early_stopping))
    txt_file_path = os.path.join(save_folder, 'loss_test_{}_{}_{}_ES_{}.txt'.format(learning_rate, learning_rate_init,
                                                                              hidden_layer_write_str,early_stopping))

    with open (txt_file_path, 'w') as f:
        for i,tuple in enumerate(write_list):
            id = str(i)
            loss = str(tuple[0])
            n_iter = str(tuple[1])
            rmse = str(tuple[2])
            avg_pc = str(tuple[3])
            polar_percent = str(tuple[4])
            random_state = str(tuple[5])

            f.write("--------------------------------------------\n".format(id))
            f.write("id: {}\n".format(id))
            f.write("loss: {}\n".format(loss))
            f.write("n_iter: {}\n".format(n_iter))
            f.write("rmse: {}\n".format(rmse))
            f.write("avg_pc: {}\n".format(avg_pc))
            f.write("polar_percent: {}\n".format(polar_percent))
            f.write("random_state: {}\n".format(random_state))
    print ("Save txt to {}".format(txt_file_path))

    with open(csv_file_path, 'w') as f:
        for i, tuple1 in enumerate(write_list):
            list1 = list(tuple1)
            list1 = [str(x) for x in list1]
            tuple_str = ','.join(list1)
            f.write(tuple_str + '\n')
    print ("Save csv to {}".format(csv_file_path))
