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
import itertools
import pickle
import numpy as np
import sys
import os
from sklearn.neural_network import MLPClassifier
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
from trade_general_funcs import compute_average_f1
from trade_general_funcs import build_hidden_layer_sizes_list
from trade_general_funcs import create_random_sub_set_list
# ==========================================================================================================




# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class MlpTradeClassifier(MlpTrade):

    def __init__(self):
        super().__init__()

        # --------------------------------------------------------------------------------------------------------------
        # container for evaluation
        # --------------------------------------------------------------------------------------------------------------
        self.accuracy_list = []
        self.average_f1_list = []
        self.label_tp_fp_tn_dict = {}
        # --------------------------------------------------------------------------------------------------------------


        # --------------------------------------------------------------------------------------------------------------
        # container for n-fold validation for different hidden layer, tp is topology, cv is cross-validation
        # --------------------------------------------------------------------------------------------------------------
        self.tp_cv_average_average_f1_list = []
        self.tp_cv_average_accuracy_list = []
        # --------------------------------------------------------------------------------------------------------------

    def set_mlp_clf(self, hidden_layer_sizes, tol=1e-6, learning_rate_init=0.001, verbose=False):
        self.hidden_size_list.append(hidden_layer_sizes)
        self.mlp_hidden_layer_sizes_list.append(hidden_layer_sizes)
        self.mlp_clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                     tol=tol, learning_rate_init=learning_rate_init,
                                     max_iter=2000, random_state=1, verbose=verbose)



    def clf_train(self, save_clsfy_path="mlp_trade_classifier", is_production=False):

        self.mlp_clf.fit(self.training_set, self.training_value_set)
        self.iteration_loss_list.append((self.mlp_clf.n_iter_, self.mlp_clf.loss_))

        # try:
        #     self.mlp_clf.fit(self.training_set, self.training_label)
        # except ValueError:
        #     print ("feature_switch_list: ", self.feature_switch_list)
        #     #logger2.error("training_set: {}".format(self.training_set))
        #     sys.exit()

        pickle.dump(self.mlp_clf, open(save_clsfy_path, "wb"))




    def clf_dev(self, save_clsfy_path="mlp_trade_classifier", is_cv=False):

        # (1.) read classifier
        mlp = pickle.load(open(save_clsfy_path, "rb"))
        #

        # (2.) get pred label list
        pred_label_list = mlp.predict(self.dev_set)
        #

        # (3.) compute the average f-measure
        pred_label_dict = collections.defaultdict(lambda: 0)
        for pred_label in pred_label_list:
            pred_label_dict[pred_label] += 1
        label_tp_fp_tn_dict = compute_average_f1(pred_label_list, self.dev_value_set)
        label_f1_list = sorted([(key, x[3]) for key, x in label_tp_fp_tn_dict.items()])
        f1_list = [x[1] for x in label_f1_list]
        average_f1 = np.average(f1_list)
        #

        # (4.) compute accuracy
        correct = 0
        for i, pred_label in enumerate(pred_label_list):
            if pred_label == self.dev_value_set[i]:
                correct += 1
        accuracy = correct / len(self.dev_value_set)
        #

        # (5.) count the occurrence for each label
        dev_label_dict = collections.defaultdict(lambda: 0)
        for dev_label in self.dev_value_set:
            dev_label_dict[dev_label] += 1
        #

        # (6.) save result for 1-fold
        self.average_f1_list.append(average_f1)
        self.accuracy_list.append(accuracy)
        #

        # print
        if not is_cv:
            print("\n=================================================================")
            print("Dev set result!")
            print("=================================================================")
            print("dev_label_dict: {}".format(list(dev_label_dict.items())))
            print("pred_label_dict: {}".format(list(pred_label_dict.items())))
            print("label_f1_list: {}".format(label_f1_list))
            print("average_f1: ", average_f1)
            print("accuracy: ", accuracy)
            print("=================================================================")
        #




    #   ====================================================================================================================
    #   CROSS VALIDATION FUNCTIONS FOR CLASSIFICATION
    #   ====================================================================================================================
    def cv_feed_and_seperate_data(self, validation_index, samples_feature_list, samples_label_list,
                                  dev_per=0.1, is_print=False):

        # clear training_set, dev_set
        self.training_set = []
        self.training_label = []
        self.dev_set = []
        self.dev_value_set = []
        #


        sample_number = len(samples_feature_list)
        sample_last_index = sample_number - 1
        dev_sample_num = math.floor(sample_number * dev_per)
        validation_split_list = [[i * dev_sample_num, (i + 1) * dev_sample_num] for i in range(10)]
        validation_split_list[-1][1] = sample_last_index

        # print ("validation_split_list: ", validation_split_list)
        dev_start_index = validation_split_list[validation_index][0]
        dev_end_index = validation_split_list[validation_index][1]

        self.training_set = samples_feature_list[0:dev_start_index] + samples_feature_list[
                                                                      dev_end_index:sample_last_index]
        self.training_label = samples_label_list[0:dev_start_index] + samples_label_list[
                                                                      dev_end_index:sample_last_index]
        self.dev_set = samples_feature_list[dev_start_index:dev_end_index]
        self.dev_value_set = samples_label_list[dev_start_index:dev_end_index]

        # count the label in traning and testing data
        dev_label_dict = collections.defaultdict(lambda: 0)
        for label in self.dev_value_set:
            dev_label_dict[label] += 1

        training_label_dict = collections.defaultdict(lambda: 0)
        for label in self.training_label:
            training_label_dict[label] += 1

        if is_print:
            print("-------------------------------------------------------------------------\n")
            print("-------------------------------------------------------------------------")
            print("Set data for validation index: {}, range: ({}, {})".format(validation_index,
                                                                              dev_start_index, dev_end_index))
            # print ("Training Label: {}".format(training_label_dict.items()))
            print("Dev Label: {}".format(dict(dev_label_dict.items())))
            print("-------------------------------------------------------------------------")

    def cv_cls_topology_test(self, input_folder, feature_switch_tuple, other_config_dict,
                             hidden_layer_config_tuple, is_random=False):
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
        dev_per = other_config_dict['dev_per']
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
        random_seed_list = other_config_dict['random_seed_list']
        # --------------------------------------------------------------------------------------------------------------

        # --------------------------------------------------------------------------------------------------------------
        # (4.) create validation_dict
        # --------------------------------------------------------------------------------------------------------------
        for random_seed in random_seed_list:
            # create the random sub set list
            dev_date_num = math.floor(len(set(date_str_list)) * dev_per)
            date_random_subset_list = \
                create_random_sub_set_list(set(date_str_list), dev_date_num, random_seed=random_seed)
            print("-----------------------------------------------------------------------------")
            print("random_seed: {}, date_random_subset_list: {}".format(random_seed, date_random_subset_list))
            self.create_train_dev_vdict(samples_feature_list, samples_value_list, date_str_list, stock_id_list,
                                        date_random_subset_list, random_seed)
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

            # random inside, make sure each date has all the
            for random_seed in random_seed_list:
                for cv_index in self.validation_dict[random_seed].keys():
                    self.rs_cv_load_train_dev_data(random_seed, cv_index)
                    self.clf_train(save_clsfy_path=clf_path)
                    self.clf_dev(save_clsfy_path=clf_path, is_cv=True)
            #

            # (c.) save the 10-cross-valiation evaluate result for each topology

            self.tp_cv_iteration_loss_list.append([x[1] for x in self.iteration_loss_list])
            self.tp_cv_average_average_f1_list.append(self.average_f1_list)
            self.tp_cv_average_accuracy_list.append(self.accuracy_list)

            # (d.) real-time print
            print("====================================================================")
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
                                            self.tp_cv_average_average_f1_list)),
                                   key=lambda x: x[-1], reverse=True)
        elif key == 'acc':
            topology_list = sorted(list(zip(self.feature_switch_list, self.feature_selected_list,
                                            self.hidden_size_list, self.tp_cv_iteration_loss_list,
                                            self.tp_cv_average_accuracy_list,
                                            self.tp_cv_average_average_f1_list)),
                                   key=lambda x: x[-2], reverse=True)
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
                f.write('----------------------------------------------------\n')
                f.write('feature_switch: {}\n'.format(feature_switch))
                f.write('feature_selected: {}\n'.format(feature_selected))
                f.write('hidden_size: {}\n'.format(hidden_size))
                f.write('average_iteration_loss: {}\n'.format(iteration_loss))
                f.write('average_accuracy: {}\n'.format(avg_accuracy))
                f.write('average_avg_f1: {}\n'.format(avg_avg_f1))

        print("Classification! Save 10-cross-validation topology test result by to {} sucessfully!".format(path))

    #   ====================================================================================================================
    #   CROSS VALIDATION FUNCTIONS FOR CLASSIFICATION END
    #   ====================================================================================================================





































