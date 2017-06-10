# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# MLP regressor only for trading, currently for a share stock and forex
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# (c) 2017 PJS, University of Sheffield, iamlxb3@gmail.com
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# ==========================================================================================================
# general import
# ==========================================================================================================
import sys
import os
import re
import math
import numpy as np
import pickle
from sklearn.neural_network import MLPRegressor
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
from trade_general_funcs import feature_degradation
from trade_general_funcs import calculate_mrse
from trade_general_funcs import get_avg_price_change
from trade_general_funcs import create_random_sub_set_list
from trade_general_funcs import list_by_index
from trade_general_funcs import build_hidden_layer_sizes_list
# ==========================================================================================================




# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class MlpTradeRegressor(MlpTrade):

    def __init__(self):
        super().__init__()

        # --------------------------------------------------------------------------------------------------------------
        # container for evaluation config and result
        # --------------------------------------------------------------------------------------------------------------
        # config
        self.include_top_list = []
        # eva
        self.mres_list = []
        self.var_std_list = []
        self.avg_price_change_list = []
        self.polar_accuracy_list = []
        # --------------------------------------------------------------------------------------------------------------


        # --------------------------------------------------------------------------------------------------------------
        # container for 1-fold validation training and dev data
        # --------------------------------------------------------------------------------------------------------------
        self.r_training_set = []
        self.r_training_value_set = []
        self.r_dev_set = []
        self.r_dev_value_set = []
        self.r_dev_date_set = []
        self.r_dev_stock_id_set = []
        # --------------------------------------------------------------------------------------------------------------


        # --------------------------------------------------------------------------------------------------------------
        # container for n-fold validation for different hidden layer, tp is topology, cv is cross-validation
        # --------------------------------------------------------------------------------------------------------------
        self.tp_cv_mres_list = []
        # (#) cv_avg_price_change_list eg. [[0.1,0.2,0.3,..0.99], ...](each list represent one topology, each element
        # (#) is 1-fold validation avg. List could be 10 or 30 long, based on random seed list).
        self.tp_cv_avg_price_change_list = []
        self.tp_cv_polar_accuracy_list = []
        # --------------------------------------------------------------------------------------------------------------


    def set_regressor(self, hidden_layer_sizes, tol=1e-8, learning_rate_init=0.001):
        self.hidden_size_list.append(hidden_layer_sizes)
        self.mlp_hidden_layer_sizes_list.append(hidden_layer_sizes)
        self.mlp_regressor = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                                          tol=tol, learning_rate_init=learning_rate_init,
                                          max_iter=1000, random_state=1)


    # ------------------------------------------------------------------------------------------------------------------
    # [C.1] read, feed and load data
    # ------------------------------------------------------------------------------------------------------------------
    def _r_feed_data(self, folder, data_per, feature_switch_tuple=None, is_random=False, random_seed=1):
        # feature_switch_tuple : (0,1,1,0,1,1,0,...) ?
        if feature_switch_tuple:
            self.feature_switch_list.append(feature_switch_tuple)
        # ::: _feed_data :::
        # TODO test the folder exists
        file_name_list = os.listdir(folder)
        file_path_list = [os.path.join(folder, x) for x in file_name_list]
        file_total_number = len(file_name_list)
        file_used_number = math.floor(data_per * file_total_number)  # restrict the number of training sample
        file_path_list = file_path_list[0:file_used_number]
        samples_feature_list = []
        samples_value_list = []
        date_str_list = []
        stock_id_list = []
        for f_path in file_path_list:
            f_name = os.path.basename(f_path)
            regression_value = float(re.findall(r'#([0-9\.\+\-e]+)#', f_name)[0])
            date_str = re.findall(r'([0-9]+-[0-9]+-[0-9]+)_', f_name)[0]
            stock_id = re.findall(r'_([0-9]{6})_', f_name)[0]
            with open(f_path, 'r') as f:
                features_list = f.readlines()[0].split(',')
                features_list = features_list[1::2]
                features_list = [float(x) for x in features_list]
                if feature_switch_tuple:
                    features_list = feature_degradation(features_list, feature_switch_tuple)
                features_array = np.array(features_list)
                # features_array = features_array.reshape(-1,1)
                samples_feature_list.append(features_array)
                samples_value_list.append(regression_value)
                date_str_list.append(date_str)
                stock_id_list.append(stock_id)
        print("read feature list and regression_value list for {} successful!".format(folder))

        # random by random seed
        if is_random:
            print("Start shuffling the data...")
            import random
            combind_list = list(zip(samples_feature_list, samples_value_list, date_str_list, stock_id_list))
            random_seed = random_seed
            random.seed(random_seed)
            random.shuffle(combind_list)
            samples_feature_list, samples_value_list, date_str_list, stock_id_list = zip(*combind_list)
            print("Data set shuffling complete! Random Seed: {}".format(random_seed))
        #

        return samples_feature_list, samples_value_list, date_str_list, stock_id_list

    def r_feed_and_separate_data(self, folder, dev_per=0.1, data_per=1.0, feature_switch_tuple=None,
                                 is_production=False, random_seed='normal'):
        '''feed and seperate data in the normal order
        '''
        # (1.) read all the data, feature customizable
        samples_feature_list, samples_value_list, \
        date_str_list, stock_id_list = self._r_feed_data(folder, data_per=data_per,
                                                         feature_switch_tuple=feature_switch_tuple, is_random=False)

        # (2.) compute the dev part index
        dev_date_num = math.floor(len(set(date_str_list)) * dev_per)
        date_random_subset_list = [set(sorted(list(set(date_str_list)))[-1 * dev_date_num:])]
        print("date_random_subset_list: ", date_random_subset_list)

        # (3.) split the data into training and developing into validation dict
        self.r_create_train_dev_vdict(samples_feature_list, samples_value_list, date_str_list, stock_id_list,
                                      date_random_subset_list, random_seed)

        # (4.) load the data for training and dev
        self.rs_cv_r_load_train_dev_data(random_seed, 0)
        print("Load train, dev data complete! Train size: {}, dev size: {}".
              format(len(self.r_training_value_set), len(self.r_dev_value_set)))


    def r_create_train_dev_vdict(self, samples_feature_list, samples_value_list,
                                 date_str_list, stock_id_list, date_random_subset_list, random_seed):
        # get the date set and count the number of unique dates

        for i, dev_date_set in enumerate(date_random_subset_list):
            all_date_set = set(date_str_list)
            training_date_set = all_date_set - dev_date_set

            # get the dev index
            dev_index_list = []
            for j, date_str in enumerate(date_str_list):
                if date_str in dev_date_set:
                    dev_index_list.append(j)
            #

            # get the training index
            training_index_list = []
            for k, date_str in enumerate(date_str_list):
                if date_str in training_date_set:
                    training_index_list.append(k)
            #

            r_training_set = list_by_index(samples_feature_list, training_index_list)
            r_training_value_set = list_by_index(samples_value_list, training_index_list)
            r_dev_set = list_by_index(samples_feature_list, dev_index_list)
            r_dev_value_set = list_by_index(samples_value_list, dev_index_list)
            r_dev_date_set = list_by_index(date_str_list, dev_index_list)
            r_dev_stock_id_set = list_by_index(stock_id_list, dev_index_list)

            self.validation_dict[random_seed][i]['r_training_set'] = r_training_set
            self.validation_dict[random_seed][i]['r_training_value_set'] = r_training_value_set
            self.validation_dict[random_seed][i]['r_dev_set'] = r_dev_set
            self.validation_dict[random_seed][i]['r_dev_value_set'] = r_dev_value_set
            self.validation_dict[random_seed][i]['r_dev_date_set'] = r_dev_date_set
            self.validation_dict[random_seed][i]['r_dev_stock_id_set'] = r_dev_stock_id_set

        validation_num = len(date_random_subset_list)
        print("Create validation_dict sucessfully! {}-fold cross validation".format(validation_num))

    def rs_cv_r_load_train_dev_data(self, random_seed, cv_index):
        self.r_training_set = self.validation_dict[random_seed][cv_index]['r_training_set']
        self.r_training_value_set = self.validation_dict[random_seed][cv_index]['r_training_value_set']
        self.r_dev_set = self.validation_dict[random_seed][cv_index]['r_dev_set']
        self.r_dev_value_set = self.validation_dict[random_seed][cv_index]['r_dev_value_set']
        self.r_dev_date_set = self.validation_dict[random_seed][cv_index]['r_dev_date_set']
        self.r_dev_stock_id_set = self.validation_dict[random_seed][cv_index]['r_dev_stock_id_set']

        # print ("Read cross validation dict for random seed {} complete! Index: {}".format(random_seed, cv_index))

    # ------------------------------------------------------------------------------------------------------------------


    # ------------------------------------------------------------------------------------------------------------------
    # [C.2] Train and Dev
    # ------------------------------------------------------------------------------------------------------------------
    def regressor_train(self, save_clsfy_path="mlp_regressor", is_cv=False, is_production=False):
        self.mlp_regressor.fit(self.r_training_set, self.r_training_value_set)
        self.iteration_loss_list.append((self.mlp_regressor.n_iter_, self.mlp_regressor.loss_))
        pickle.dump(self.mlp_regressor, open(save_clsfy_path, "wb"))

        if is_production:
            print("classifier for production saved to {} successfully!".format(save_clsfy_path))


    def regressor_dev(self, save_clsfy_path="mlp_regressor", is_cv=False, include_top_list=[1]):
        mlp_regressor = pickle.load(open(save_clsfy_path, "rb"))
        pred_value_list = np.array(mlp_regressor.predict(self.r_dev_set))
        actual_value_list = np.array(self.r_dev_value_set)
        mrse = calculate_mrse(actual_value_list, pred_value_list)
        date_list = self.r_dev_date_set
        stock_id_list = self.r_dev_stock_id_set
        avg_price_change_tuple, var_tuple, std_tuple = get_avg_price_change(pred_value_list, actual_value_list,
                                                                                  date_list, stock_id_list,
                                                                                  include_top_list=
                                                                                  include_top_list)

        # count how many predicted value has the same polarity as actual value
        polar_list = [1 for x, y in zip(pred_value_list, actual_value_list) if x * y >= 0]
        polar_count = len(polar_list)
        polar_percent = polar_count / len(pred_value_list)
        #

        self.mres_list.append(mrse)
        self.avg_price_change_list.append(avg_price_change_tuple)
        self.polar_accuracy_list.append(polar_percent)
        self.var_std_list.append((var_tuple, std_tuple))

        # <uncomment for debugging>
        if not is_cv:
            print("----------------------------------------------------------------------------------------")
            print("actual_value_list, ", actual_value_list)
            print("pred_value_list, ", pred_value_list)
            print("polarity: {}".format(polar_percent))
            print("mrse: {}".format(mrse))
            print("avg_price_change: {}".format(avg_price_change_tuple))
            print("----------------------------------------------------------------------------------------")
        else:
            pass
            # print("Testing complete! Testing Set size: {}".format(len(self.r_dev_value_set)))
            # <uncomment for debugging>

    # ------------------------------------------------------------------------------------------------------------------



    # ------------------------------------------------------------------------------------------------------------------
    # [C.3] Topology Test
    # ------------------------------------------------------------------------------------------------------------------
    def cv_r_topology_test(self, input_folder, feature_switch_tuple, other_config_dict,
                           hidden_layer_config_tuple):
        '''10 cross validation test for mlp regressor'''


        # :::topology_test:::


        # ==============================================================================================================
        # Cross Validation Train And Test
        # ==============================================================================================================

        # (1.) read the whole data set
        # cut the number of training sample
        # create list under different random seed
        random_seed_list = other_config_dict['random_seed_list']
        date_per = other_config_dict['date_per']
        dev_per = other_config_dict['dev_per']
        samples_feature_list, samples_value_list, \
        date_str_list, stock_id_list = self._r_feed_data(input_folder, data_per=date_per,
                                                         feature_switch_tuple=feature_switch_tuple,
                                                         is_random=False)
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
        include_top_list = other_config_dict['include_top_list']
        self.include_top_list = include_top_list

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
            self.r_create_train_dev_vdict(samples_feature_list, samples_value_list, date_str_list, stock_id_list,
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
            self.mres_list = []
            self.avg_price_change_list = []
            self.polar_accuracy_list = []
            self.var_std_list = []

            # (b.) 10-cross-validation train and test
            self.set_regressor(hidden_layer_sizes, learning_rate_init=learning_rate_init, tol=tol)

            # random inside, make sure each date has all the
            for random_seed in random_seed_list:
                for cv_index in self.validation_dict[random_seed].keys():
                    self.rs_cv_r_load_train_dev_data(random_seed, cv_index)
                    self.regressor_train(save_clsfy_path=clf_path, is_cv=True)
                    self.regressor_dev(save_clsfy_path=clf_path, is_cv=True, include_top_list=include_top_list)
            #

            # (c.) save the 10-cross-valiation evaluate result for each topology
            self.tp_cv_iteration_loss_list.append(self.iteration_loss_list)
            self.tp_cv_mres_list.append(self.mres_list)
            self.tp_cv_avg_price_change_list.append(self.avg_price_change_list)
            self.tp_cv_polar_accuracy_list.append(self.polar_accuracy_list)

            # TODO ignore var and std for a while
            # self.cv_var_std_list.append(self.var_std_list)
            #


            # (d.) real-time print
            print("====================================================================")
            print("Feature selected: {}, Total number: {}".format(self.feature_selected_list[-1],
                                                                  self.feature_switch_list[-1].count(1)))
            print("Average mres: {} len: {}".format(np.average(self.mres_list), len(self.mres_list)))
            for j, top in enumerate(include_top_list):
                n_top_list = [x[j] for x in self.avg_price_change_list]
                print("Top: {} Average price change: {}".format(top, np.average(n_top_list)))

            # TODO ignore var,std for a while
            # print ("Average var: {}, Average std: {}".format(np.average([x[0] for x in self.var_std_list]),
            #                                  np.average([x[1] for x in self.var_std_list])))

            print("Average polarity: {}".format(np.average(self.polar_accuracy_list)))
            print("Average iteration_loss: {}".format(np.average([x[1] for x in self.iteration_loss_list])))
            print("====================================================================")
            print("Completeness: {:.5f}".format((i + 1) / hidden_layer_sizes_combination))
            print("====================================================================")

            if i != 0 and i % 10 == 0:
                self._r_print_real_time_best_result()


    def cv_r_save_feature_topology_result(self, path, key='mres'):
        # compute the average for each list
        cv_iteration_loss_list = [[y[1] for y in x] for x in self.tp_cv_iteration_loss_list]
        cv_iteration_loss_list = [np.average(x) for x in cv_iteration_loss_list]
        cv_polar_accuracy_list = [np.average(x) for x in self.tp_cv_polar_accuracy_list]
        # cv_avg_price_change_list = [np.average(x) for x in self.cv_avg_price_change_list]
        cv_mres_list = [np.average(x) for x in self.tp_cv_mres_list]

        topology_list = list(zip(self.feature_switch_list, self.feature_selected_list,
                                 self.hidden_size_list, cv_iteration_loss_list,
                                 cv_polar_accuracy_list, cv_mres_list))

        if key == 'mres':
            topology_list = sorted(topology_list,
                                   key=lambda x: x[-1])
            # write to file
            with open(path, 'w', encoding='utf-8') as f:
                for tuple1 in topology_list:
                    feature_switch = str(tuple1[0])
                    feature_selected = str(tuple1[1])
                    hidden_size = str(tuple1[2])
                    iteration_loss = str(tuple1[3])
                    polar_accuracy = str(tuple1[4])
                    mres = str(tuple1[5])
                    # TODO ignore var and std for a while
                    # var = str(tuple1[7])
                    # std = str(tuple1[8])
                    f.write('----------------------------------------------------\n')
                    f.write('feature_switch: {}\n'.format(feature_switch))
                    f.write('feature_selected: {}\n'.format(feature_selected))
                    f.write('hidden_size: {}\n'.format(hidden_size))
                    f.write('average_iteration_loss: {}\n'.format(iteration_loss))
                    f.write('average_polar_accuracy: {}\n'.format(polar_accuracy))
                    f.write('average_mres: {}\n'.format(mres))
                    print("Regression! Save 10-cross-validation topology test result by {} to {} sucessfully!".format(
                        key, path))
                    #

        elif key == 'polar':
            topology_list = sorted(topology_list,
                                   key=lambda x: x[-2], reverse=True)
            # write to file
            with open(path, 'w', encoding='utf-8') as f:
                for tuple1 in topology_list:
                    feature_switch = str(tuple1[0])
                    feature_selected = str(tuple1[1])
                    hidden_size = str(tuple1[2])
                    iteration_loss = str(tuple1[3])
                    polar_accuracy = str(tuple1[4])
                    mres = str(tuple1[5])
                    # TODO ignore var and std for a while
                    # var = str(tuple1[7])
                    # std = str(tuple1[8])
                    f.write('----------------------------------------------------\n')
                    f.write('feature_switch: {}\n'.format(feature_switch))
                    f.write('feature_selected: {}\n'.format(feature_selected))
                    f.write('hidden_size: {}\n'.format(hidden_size))
                    f.write('average_iteration_loss: {}\n'.format(iteration_loss))
                    f.write('average_polar_accuracy: {}\n'.format(polar_accuracy))
                    f.write('average_mres: {}\n'.format(mres))
                    print("Regression! Save 10-cross-validation topology test result by {} to {} sucessfully!".format(
                        key, path))
                    #

        elif key == 'avg_pc':
            # write the best result under different trading strategy
            for i, include_top in enumerate(self.include_top_list):

                # get the cv_avg_price_change_list for a particular strategy
                top_n_pc_list = [[y[i] for y in x] for x in self.tp_cv_avg_price_change_list]
                cv_avg_price_change_list = [np.average(x) for x in top_n_pc_list]
                topology_list = list(zip(self.feature_switch_list, self.feature_selected_list,
                                         self.hidden_size_list, cv_iteration_loss_list,
                                         cv_polar_accuracy_list, cv_avg_price_change_list, cv_mres_list))
                topology_list = sorted(topology_list,
                                       key=lambda x: x[-2], reverse=True)
                #

                # modify the path according to how many top N stocks are traded each week

                upper_folder = os.path.dirname(path)
                path_base_name = os.path.basename(path)[:-4]
                new_name = path_base_name + "_top_{}.txt".format(include_top)
                new_path = os.path.join(upper_folder, new_name)
                #

                # save file
                with open(new_path, 'w', encoding='utf-8') as f:
                    for tuple1 in topology_list:
                        feature_switch = str(tuple1[0])
                        feature_selected = str(tuple1[1])
                        hidden_size = str(tuple1[2])
                        iteration_loss = str(tuple1[3])
                        polar_accuracy = str(tuple1[4])
                        avg_price_change = str(tuple1[5])
                        mres = str(tuple1[6])
                        # TODO ignore var and std for a while
                        # var = str(tuple1[7])
                        # std = str(tuple1[8])
                        f.write('----------------------------------------------------\n')
                        f.write('feature_switch: {}\n'.format(feature_switch))
                        f.write('feature_selected: {}\n'.format(feature_selected))
                        f.write('hidden_size: {}\n'.format(hidden_size))
                        f.write('average_iteration_loss: {}\n'.format(iteration_loss))
                        f.write('average_polar_accuracy: {}\n'.format(polar_accuracy))
                        f.write('average_avg_price_change: {}\n'.format(avg_price_change))
                        f.write('average_mres: {}\n'.format(mres))
                        # # TODO ignore var and std for a while
                        print(
                            "Regression! Save 10-cross-validation topology test result by {} to {} sucessfully!".format(
                                key, new_path))
                        # save file

        else:
            print("Key should be mres or avg_pc, key: {}".format(key))

            # ==============================================================================================================
            # avg_pc
            # ==============================================================================================================



            # # TODO ignore var and std for a while
            # _var_list = [[y[0] for y in x] for x in self.cv_var_std_list] # cv_var_std_list : [[(0.1,0.2), ...], [(0.3,0.4), ...], ...]
            # _var_list = [np.average(x) for x in _var_list]
            # _std_list = [[y[1] for y in x] for x in self.cv_var_std_list] # cv_var_std_list : [[(0.1,0.2), ...], [(0.3,0.4), ...], ...]
            # _std_list = [np.average(x) for x in _std_list]
            #

            # ==============================================================================================================


    def _r_print_real_time_best_result(self):
        for i, include_top in enumerate(self.include_top_list):
            # get the cv_avg_price_change_list for a particular strategy
            top_n_pc_list = [[y[i] for y in x] for x in self.tp_cv_avg_price_change_list]
            cv_avg_price_change_list = [np.average(x) for x in top_n_pc_list]

            topology_list = list(zip(self.feature_switch_list, self.feature_selected_list,
                                     self.hidden_size_list, cv_avg_price_change_list))
            sorted_topology_list = sorted(topology_list,
                                          key=lambda x: x[-1], reverse=True)

            top_feature_switch = sorted_topology_list[0][0]
            top_hidden_size = sorted_topology_list[0][2]
            top_cv_avg_price_change = sorted_topology_list[0][3]

            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print("{}-TOP-BEST".format(include_top))
            print("cv_avg_price_change: ", top_cv_avg_price_change)
            print("feature_switch: ", top_feature_switch)
            print("hidden_size: ", top_hidden_size)

    # ------------------------------------------------------------------------------------------------------------------

