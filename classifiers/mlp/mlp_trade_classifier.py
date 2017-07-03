# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# MLP classifier only for trading, currently for a share stock and forex
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# (c) 2017 PJS, University of Sheffield, iamlxb3@gmail.com
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# ==========================================================================================================
# general import
# ==========================================================================================================
import math
import collections
import numpy as np
import sys
import os
# ==========================================================================================================


# ==========================================================================================================
# ADD SYS PATH
# ==========================================================================================================
parent_folder = os.path.dirname((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
path1 = os.path.join(parent_folder, 'general_functions')
path2 = os.path.join(parent_folder, 'strategy')
sys.path.append(path1)
sys.path.append(path2)
# ==========================================================================================================


# ==========================================================================================================
# local package import
# ==========================================================================================================
from mlp_trade import MlpTrade
from mlp_classifier import MlpClassifier_P
from trade_general_funcs import build_hidden_layer_sizes_list
from trade_general_funcs import create_random_sub_set_list
# ==========================================================================================================




# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class MlpTradeClassifier(MlpTrade, MlpClassifier_P):

    def __init__(self):
        super().__init__()


    #   ====================================================================================================================
    #   CROSS VALIDATION FUNCTIONS FOR CLASSIFICATION
    #   ====================================================================================================================
    def cv_cls_topology_test(self, input_folder, feature_switch_tuple, other_config_dict,
                             hidden_layer_config_tuple, is_random=False, is_window_shift = False):
        '''10 cross validation test for mlp classifier'''
        # :::topology_test:::


        # ==============================================================================================================
        # Cross Validation Train And Test
        # ==============================================================================================================

        # feature switch tuple
        if feature_switch_tuple:
            self.feature_switch_list.append(feature_switch_tuple)

        # (1.) read the whole data set
        # cut the number of training sample
        data_per = other_config_dict['data_per']
        samples_feature_list, samples_value_list, date_str_list, stock_id_list = \
            self._feed_data(input_folder, data_per=data_per,
                                                       feature_switch_tuple=feature_switch_tuple,
                                                       is_random=False, mode = 'clf')
        # --------------------------------------------------------------------------------------------------------------

        # (2.) construct hidden layer size list
        hidden_layer_sizes_list = build_hidden_layer_sizes_list(hidden_layer_config_tuple)
        hidden_layer_sizes_combination = len(hidden_layer_sizes_list)
        print("Total {} hidden layer size combination to test".format(hidden_layer_sizes_combination))
        # --------------------------------------------------------------------------------------------------------------

        # (3.) set MLP parameters
        learning_rate_init = other_config_dict['learning_rate_init']
        clf_path = other_config_dict['clf_path']
        tol = other_config_dict['tol']

        # (4.) Data Pre-processing
        is_standardisation = other_config_dict['is_standardisation']
        is_PCA = other_config_dict['is_PCA']
        pca_n_component = other_config_dict['pca_n_component']

        # (5.) Data split mode
        if not is_window_shift:
            random_seed_list = other_config_dict['random_seed_list']
        if is_window_shift:
            shifting_size_percent = other_config_dict['shifting_size_percent']
            shift_num = other_config_dict['shift_num']

        # (6.) test for the size of the traning set
        training_set_percent = other_config_dict['training_set_percent']
        # --------------------------------------------------------------------------------------------------------------

        # --------------------------------------------------------------------------------------------------------------
        # (4.) create validation_dict
        # --------------------------------------------------------------------------------------------------------------
        if is_window_shift:
            self.create_train_dev_vdict_window_shift(samples_feature_list, samples_value_list,
                           date_str_list, stock_id_list, is_cv=True, shifting_size_percent = shifting_size_percent,
                                                     shift_num = shift_num,
                                                     is_standardisation = is_standardisation, is_PCA = is_PCA,
                                                     pca_n_component = pca_n_component, training_set_percent = training_set_percent)
        else:
            dev_per = other_config_dict['dev_per']
            for random_seed in random_seed_list:
                # create the random sub set list
                dev_date_num = math.floor(len(set(date_str_list)) * dev_per)
                date_random_subset_list = \
                    create_random_sub_set_list(set(date_str_list), dev_date_num, random_seed=random_seed)
                print("-----------------------------------------------------------------------------")
                print("random_seed: {}, date_random_subset_list: {}".format(random_seed, date_random_subset_list))
                self.create_train_dev_vdict_stock(samples_feature_list, samples_value_list, date_str_list, stock_id_list,
                                                  date_random_subset_list, random_seed,
                                                  is_standardisation = is_standardisation, is_PCA = is_PCA,
                                                  pca_n_component = pca_n_component)
        # --------------------------------------------------------------------------------------------------------------

        # (4.) test the performance of different topology of MLP by 10-cross validation
        for i, hidden_layer_sizes in enumerate(hidden_layer_sizes_list):
            print("====================================================================")
            print("Topology: {} starts training and testing".format(hidden_layer_sizes))
            print("====================================================================")

            self._update_feature_switch_list(i)

            # (a.) clear the evaluation list for one hidden layer topology
            self.iteration_loss_list = []
            self.average_f1_list = []
            self.accuracy_list = []
            #

            # (b.) train and dev
            self.set_mlp_clf(hidden_layer_sizes, learning_rate_init=learning_rate_init, tol=tol)

            if is_window_shift:
                # (b.1) [window-shifting-n-fold]
                random_seed = 'window_shift'
                for shift in self.validation_dict[random_seed].keys():
                    self.trade_rs_cv_load_train_dev_data(random_seed, shift)
                    self.clf_train(save_clsfy_path=clf_path)
                    self.clf_dev(save_clsfy_path=clf_path, is_cv=True)
            else:
                # (b.2) [random-n-fold]-random inside, make sure each date has all the
                for random_seed in random_seed_list:
                    for cv_index in self.validation_dict[random_seed].keys():
                        self.trade_rs_cv_load_train_dev_data(random_seed, cv_index)
                        self.clf_train(save_clsfy_path=clf_path)
                        self.clf_dev(save_clsfy_path=clf_path, is_cv=True)
            #

            # (c.) save the 10-cross-valiation evaluate result for each topology

            self.tp_cv_iteration_loss_list.append([x[1] for x in self.iteration_loss_list])
            self.tp_cv_average_average_f1_list.append(self.average_f1_list)
            self.tp_cv_average_accuracy_list.append(self.accuracy_list)
            self.tp_cv_pca_n_component_list.append(pca_n_component)


            # (d.) real-time print
            print("====================================================================")
            print("Feature selected: {}, Total number: {}".format(self.feature_selected_list[-1],
                                                                  self.feature_switch_list[-1].count(1)))
            print("PCA-n-component: {}".format(pca_n_component))
            print("Average avg f1: {}".format(np.average(self.average_f1_list)))
            print("Average accuracy: {}".format(np.average(self.accuracy_list)))
            print("Average iteration_loss: {}".format(np.average([x[1] for x in self.iteration_loss_list])))
            print("====================================================================")
            print("Completeness: {:.5f}".format((i + 1) / hidden_layer_sizes_combination))
            print("====================================================================")
            #
            # ==============================================================================================================
            # Cross Validation Train And Test END
            # ==============================================================================================================
            if i != 0 and i % 10 == 0:
                self._c_print_real_time_best_result()
            # --------------------------------------------------------------------------------------------------------------

    def cv_cls_save_feature_topology_result(self, path, key = 'f_m'):

        # compute the average for each list
        self.tp_cv_iteration_loss_list = [np.average(x) for x in self.tp_cv_iteration_loss_list]
        self.tp_cv_average_accuracy_list = [np.average(x) for x in self.tp_cv_average_accuracy_list]
        self.tp_cv_average_average_f1_list = [np.average(x) for x in self.tp_cv_average_average_f1_list]


        if key == 'f_m':
            topology_list = sorted(list(zip(self.feature_switch_list, self.feature_selected_list,
                                            self.hidden_size_list, self.tp_cv_iteration_loss_list,
                                            self.tp_cv_average_accuracy_list,
                                            self.tp_cv_average_average_f1_list,
                                            self.tp_cv_pca_n_component_list)),
                                   key=lambda x: x[-2], reverse=True)
        elif key == 'acc':
            topology_list = sorted(list(zip(self.feature_switch_list, self.feature_selected_list,
                                            self.hidden_size_list, self.tp_cv_iteration_loss_list,
                                            self.tp_cv_average_accuracy_list,
                                            self.tp_cv_average_average_f1_list,
                                            self.tp_cv_pca_n_component_list)),
                                   key=lambda x: x[-3], reverse=True)
        else:
            print ("Please type the right key!")
            sys.exit()

        with open(path, 'w', encoding='utf-8') as f:
            for tuple1 in topology_list:
                feature_switch = str(tuple1[0])
                feature_selected = str(tuple1[1])
                hidden_size = str(tuple1[2])
                iteration_loss = str(tuple1[3])
                avg_accuracy = str(tuple1[4])
                avg_avg_f1 = str(tuple1[5])
                pca_n_component = str(tuple1[6])
                f.write('----------------------------------------------------\n')
                f.write('feature_switch: {}\n'.format(feature_switch))
                f.write('feature_selected: {}\n'.format(feature_selected))
                f.write('hidden_size: {}\n'.format(hidden_size))
                f.write('average_iteration_loss: {}\n'.format(iteration_loss))
                f.write('average_accuracy: {}\n'.format(avg_accuracy))
                f.write('average_avg_f1: {}\n'.format(avg_avg_f1))
                f.write('pca_n_component: {}\n'.format(pca_n_component))


        print("Classification! Save 10-cross-validation topology test result by to {} sucessfully!".format(path))

    #   ====================================================================================================================
    #   CROSS VALIDATION FUNCTIONS FOR CLASSIFICATION END
    #   ====================================================================================================================

    def _c_print_real_time_best_result(self):
        # get the cv_avg_price_change_list for a particular strategy
        self.tp_cv_iteration_loss_list = [np.average(x) for x in self.tp_cv_iteration_loss_list]
        self.tp_cv_average_accuracy_list = [np.average(x) for x in self.tp_cv_average_accuracy_list]
        self.tp_cv_average_average_f1_list = [np.average(x) for x in self.tp_cv_average_average_f1_list]

        sorted_topology_list = sorted(list(zip(self.feature_switch_list, self.feature_selected_list,
                                        self.hidden_size_list, self.tp_cv_iteration_loss_list,
                                        self.tp_cv_average_accuracy_list,
                                        self.tp_cv_average_average_f1_list,
                                        self.tp_cv_pca_n_component_list)),
                               key=lambda x: x[-2], reverse=True)

        top_feature_switch = sorted_topology_list[0][0]
        top_hidden_size = sorted_topology_list[0][2]
        top_iteration_loss = sorted_topology_list[0][3]
        top_accuracy = sorted_topology_list[0][4]
        top_f1 = sorted_topology_list[0][5]
        top_pca_n_component = sorted_topology_list[0][6]

        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("BEST RESULT BY AVERAGE F-MEASURE SO FAR!")
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("feature_switch: ", top_feature_switch)
        print("hidden_size: ", top_hidden_size)
        print("iteration_loss: ", top_iteration_loss)
        print("accuracy: ", top_accuracy)
        print("f-measure: ", top_f1)
        print("top_pca_n_component: ", top_pca_n_component)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")


