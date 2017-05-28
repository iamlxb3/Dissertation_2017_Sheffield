from sklearn.neural_network import MLPClassifier, MLPRegressor
import os
import numpy as np
import collections
import re
import pickle
import math
import datetime
import time
import itertools
import sys
from pjslib.logger import logger2

# ==========================================================================================================
# ADD SYS PATH
# ==========================================================================================================
parent_folder = os.path.dirname((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
path1 = os.path.join(parent_folder, 'general_functions')
sys.path.append(path1)
# ==========================================================================================================

# ==========================================================================================================
# local package import
# ==========================================================================================================
from general_funcs import calculate_mrse
# ==========================================================================================================

class MlpClassifier:
    def __init__(self):
        self.mlp_hidden_layer_sizes_list = []
        self.training_set = []
        self.training_label = []
        self.dev_set = []
        self.dev_label = []
        self.test_set = []
        self.test_label = []
        self.hidden_size_list = []
        self.accuracy_list = []
        self.average_f1_list = []
        self.label_tp_fp_tn_dict = {}
        self.feature_switch_list = []
        self.feature_selected_list = []
        self.iteration_loss_list = []
        self.mres_list = []
        self.avg_price_change_list = []
        self.polar_accuracy_list = []

        # cross validation list for classification
        self.cv_average_average_f1_list = []
        self.cv_average_accuracy_list = []
        #

        # cross validation list for regression
        self.cv_mres_list = []
        self.cv_avg_price_change_list = []
        self.cv_polar_accuracy_list = []
        #

        # for all
        self.cv_iteration_loss_list = []
        #

    def count_label(self, folder):
        file_name_list = os.listdir(folder)
        label_dict = collections.defaultdict(lambda: 0)
        for file_name in file_name_list:
            try:
                label = re.findall(r'_([0-9A-Za-z]+)\.', file_name)[0]
            except IndexError:
                print("Check folder path!")
                break
            label_dict[label] += 1
        print("label_dict: {}".format(list(label_dict.items())))

    def set_mlp(self, hidden_layer_sizes, tol=1e-6, learning_rate_init=0.001, verbose = False):
        self.hidden_size_list.append(hidden_layer_sizes)
        self.mlp_hidden_layer_sizes_list.append(hidden_layer_sizes)
        # self.mlp_clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
        #                              tol = tol, learning_rate_init = learning_rate_init, verbose = True,
        #                              solver = 'sgd', momentum = 0.3,  max_iter = 10000)
        self.mlp_clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                     tol=tol, learning_rate_init=learning_rate_init,
                                     max_iter=2000, random_state=1, verbose = verbose)

    def _feature_degradation(self, features_list, feature_switch_tuple):
        new_feature_list = []
        for i, switch in enumerate(feature_switch_tuple):
            if switch == 1:
                new_feature_list.append(features_list[i])
        return new_feature_list

    def _feed_data(self, folder, data_per, feature_switch_tuple=None, is_random = False):
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
        samples_label_list = []
        for f_path in file_path_list:
            f_name = os.path.basename(f_path)
            label = re.findall(r'_([A-Za-z0-9]+)\.', f_name)[0]
            with open(f_path, 'r') as f:
                features_list = f.readlines()[0].split(',')
                features_list = features_list[1::2]
                features_list = [float(x) for x in features_list]
                if feature_switch_tuple:
                    features_list = self._feature_degradation(features_list, feature_switch_tuple)
                features_array = np.array(features_list)
                # features_array = features_array.reshape(-1,1)
                samples_feature_list.append(features_array)
                samples_label_list.append(label)
        print("read feature list and label list for {} successful!".format(folder))

        # random by random seed
        if is_random:
            print("Start shuffling the data...")
            import random
            combind_list = list(zip(samples_feature_list, samples_label_list))
            random_seed = 1
            random.seed(random_seed)
            random.shuffle(combind_list)
            samples_feature_list, samples_label_list = zip(*combind_list)
            print ("Data set shuffling complete! Random Seed: {}".format(random_seed))
        #
        return samples_feature_list, samples_label_list

    def read_selected_feature_list(self, folder, feature_switch_list):
        file_name_list = os.listdir(folder)
        file_path_0 = [os.path.join(folder, x) for x in file_name_list][0]
        with open(file_path_0, 'r', encoding='utf-8') as f:
            feature_name_list = f.readlines()[0].split(',')[::2]
            selected_feature_list = self._feature_degradation(feature_name_list, feature_switch_list)
        self.feature_selected_list.append(selected_feature_list)

    def feed_and_seperate_data(self, folder, dev_per=0.2, data_per=1.0, feature_switch_tuple=None):

        # clear training_set, dev_set
        self.training_set = []
        self.training_label = []
        self.dev_set = []
        self.dev_label = []
        #

        samples_dict = collections.defaultdict(lambda: [])

        # cut the number of training sample
        samples_feature_list, samples_label_list = self._feed_data(folder, data_per,
                                                                   feature_switch_tuple=feature_switch_tuple)

        for i, label in enumerate(samples_label_list):
            samples_dict[label].append(samples_feature_list[i])

        for label, feature_list in samples_dict.items():
            sample_number = len(feature_list)
            label_list = [label for x in range(sample_number)]
            dev_sample_num = math.floor(sample_number * dev_per) * -1
            self.training_set.extend(feature_list[0:dev_sample_num])
            self.training_label.extend(label_list[0:dev_sample_num])
            self.dev_set.extend(feature_list[dev_sample_num:])
            self.dev_label.extend(label_list[dev_sample_num:])

        dev_label_dict = collections.defaultdict(lambda: 0)
        for label in self.dev_label:
            dev_label_dict[label] += 1

        training_label_dict = collections.defaultdict(lambda: 0)
        for label in self.training_label:
            training_label_dict[label] += 1

        print("dev_label_dict: ", list(dev_label_dict.items()))
        print("training_label_dict: ", list(training_label_dict.items()))




    def train(self, save_clsfy_path="mlp_classifier", is_cv = False):

        self.mlp_clf.fit(self.training_set, self.training_label)
        self.iteration_loss_list.append((self.mlp_clf.n_iter_, self.mlp_clf.loss_))

        # try:
        #     self.mlp_clf.fit(self.training_set, self.training_label)
        # except ValueError:
        #     print ("feature_switch_list: ", self.feature_switch_list)
        #     #logger2.error("training_set: {}".format(self.training_set))
        #     sys.exit()

        pickle.dump(self.mlp_clf, open(save_clsfy_path, "wb"))

    def _compute_average_f1(self, pred_label_list, gold_label_list):
        label_tp_fp_tn_dict = collections.defaultdict(lambda: [0, 0, 0, 0])  # tp,fp,fn,f1
        label_set = set(gold_label_list)

        for i, pred_label in enumerate(pred_label_list):
            gold_label = gold_label_list[i]
            for label in label_set:
                if pred_label == label and gold_label == label:
                    label_tp_fp_tn_dict[label][0] += 1  # true positve
                elif pred_label == label and gold_label != label:
                    label_tp_fp_tn_dict[label][1] += 1  # false positve
                elif pred_label != label and gold_label == label:
                    label_tp_fp_tn_dict[label][2] += 1  # false nagative

        # compute f1
        for label, f1_list in label_tp_fp_tn_dict.items():
            tp, fp, fn = f1_list[0:3]
            if tp + fp == 0:
                precision = 0.0
            else:
                precision = tp / (tp + fp)

            recall = tp / (tp + fn)
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = (2 * precision * recall) / (precision + recall)  # equal weight to precision and recall
            f1_list[3] = f1
            # reference:
            # https://www.quora.com/What-is-meant-by-F-measure-Weighted-F-Measure-and-Average-F-Measure-in-NLP-Evaluation

        return label_tp_fp_tn_dict

    def dev(self, save_clsfy_path="mlp_classifier", is_cv = False):
        mlp = pickle.load(open(save_clsfy_path, "rb"))

        pred_label_list = mlp.predict(self.dev_set)

        pred_label_dict = collections.defaultdict(lambda: 0)
        for pred_label in pred_label_list:
            pred_label_dict[pred_label] += 1

        label_tp_fp_tn_dict = self._compute_average_f1(pred_label_list, self.dev_label)
        self.label_tp_fp_tn_dict = label_tp_fp_tn_dict
        label_f1_list = sorted([(key, x[3]) for key, x in label_tp_fp_tn_dict.items()])
        f1_list = [x[1] for x in label_f1_list]
        average_f1 = np.average(f1_list)
        self.average_f1_list.append(average_f1)


        correct = 0
        for i, pred_label in enumerate(pred_label_list):
            if pred_label == self.dev_label[i]:
                correct += 1

        accuracy = correct/len(self.dev_label)
        self.accuracy_list.append(accuracy)

        dev_label_dict = collections.defaultdict(lambda: 0)
        for dev_label in self.dev_label:
            dev_label_dict[dev_label] += 1

        # print("\n=================================================================")
        # print("Dev set result!")
        # print("=================================================================")
        # print("dev_label_dict: {}".format(list(dev_label_dict.items())))
        # print("pred_label_dict: {}".format(list(pred_label_dict.items())))
        # print("label_f1_list: {}".format(label_f1_list))
        # print("average_f1: ", average_f1)
        # print("accuracy: ", accuracy)
        # print("=================================================================")

    def weekly_predict(self, input_folder, classifier_path, prediction_save_path):
        mlp = pickle.load(open(classifier_path, "rb"))

        file_name_list = os.listdir(input_folder)
        prediction_set = []

        # find the nearest date
        date_set = set()
        for file_name in file_name_list:
            date = re.findall(r'([0-9]+-[0-9]+-[0-9]+)_', file_name)[0]
            date_set.add(date)
        nearest_date = sorted(list(date_set), reverse=True)[0]
        nearest_date_temp = time.strptime(nearest_date, '%Y-%m-%d')
        nearest_date_obj = datetime.datetime(*nearest_date_temp[:3])

        print("==============================================================================")
        print("Prediction complete! This prediction is based on the data of DATE:[{}]".format(nearest_date))
        print("==============================================================================")

        if nearest_date_obj.weekday() != 4:
            print("WARNING! The nearest date for prediction is not friday!")

        # accumulate the prediction result
        for file_name in file_name_list:

            stock_id = re.findall(r'_([0-9]+).txt', file_name)[0]
            date = re.findall(r'([0-9]+-[0-9]+-[0-9]+)_', file_name)[0]
            if date != nearest_date:
                continue
            file_path = os.path.join(input_folder, file_name)

            with open(file_path, 'r') as f:
                feature_value_list = f.readlines()[0].strip().split(',')[1::2]
                feature_value_list = [float(x) for x in feature_value_list]
                # =================================================================================================
                # construct features_set and predict
                # =================================================================================================
                feature_array = np.array(feature_value_list).reshape(1, -1)
                pred_value = float(mlp.predict(feature_array)[0])
                prediction_set.append((stock_id, pred_value))
                # =================================================================================================

        # write the prediciton result to file
        prediction_set = sorted(prediction_set, key=lambda x: x[1], reverse=True)
        with open(prediction_save_path, 'w', encoding='utf-8') as f:
            for stock_id, pred_value in prediction_set:
                f.write(stock_id + ' ' + str(pred_value) + '\n')
        #

        print("Prediction result save to {} successful!".format(prediction_save_path))

    def topology_test(self, other_config_dict, hidden_layer_config_tuple):

        def _build_hidden_layer_sizes_list(hidden_layer_config_tuple):
            hidden_layer_node_min, hidden_layer_node_max, hidden_layer_node_step, hidden_layer_depth_min, \
            hidden_layer_depth_max = hidden_layer_config_tuple

            hidden_layer_unit_list = [x for x in range(hidden_layer_node_min, hidden_layer_node_max + 1)]
            hidden_layer_unit_list = hidden_layer_unit_list[::hidden_layer_node_step]
            #

            hidden_layer_layer_list = [x for x in range(hidden_layer_depth_min, hidden_layer_depth_max + 1)]
            #
            hidden_layer_sizes_list = list(itertools.product(hidden_layer_unit_list, hidden_layer_layer_list))
            return hidden_layer_sizes_list

        # :::topology_test:::




        hidden_layer_sizes_list = _build_hidden_layer_sizes_list(hidden_layer_config_tuple)
        learning_rate_init = other_config_dict['learning_rate_init']
        clf_path = other_config_dict['clf_path']
        tol = other_config_dict['tol']

        for i, hidden_layer_sizes in enumerate(hidden_layer_sizes_list):
            # _update_feature_switch_list
            self._update_feature_switch_list(i)

            self.set_mlp(hidden_layer_sizes, learning_rate_init=learning_rate_init, tol=tol)
            self.train(save_clsfy_path=clf_path)
            self.dev(save_clsfy_path=clf_path)

            # self.save_feature_topology_result(topology_result_path)

    def _update_feature_switch_list(self, i):
        if i != 0:
            # --------------------------------------------------------------------------
            # update feature_switch_list and feature_selected list for easy output
            # --------------------------------------------------------------------------
            self.feature_switch_list.append(self.feature_switch_list[-1])
            self.feature_selected_list.append(self.feature_selected_list[-1])
            # --------------------------------------------------------------------------

    def generate_feature_switch_list(self, folder):
        # read feature length
        file_name_list = os.listdir(folder)
        file_path_0 = [os.path.join(folder, x) for x in file_name_list][0]
        with open(file_path_0, 'r', encoding='utf-8') as f:
            feature_name_list = f.readlines()[0].split(',')[::2]
        feature_num = len(feature_name_list)
        feature_switch_list_all = list(itertools.product([0, 1], repeat=feature_num))
        feature_switch_list_all.remove(tuple([0 for x in range(feature_num)]))
        print("Total feature combination: {}".format(len(feature_switch_list_all)))
        return feature_switch_list_all

    def save_feature_topology_result(self, path):
        topology_list = sorted(list(zip(self.feature_switch_list, self.feature_selected_list,
                                        self.hidden_size_list, self.average_f1_list, self.iteration_loss_list)),
                               key=lambda x: x[3], reverse=True)
        with open(path, 'w', encoding='utf-8') as f:
            for tuple1 in topology_list:
                feature_switch_list = str(tuple1[0])
                feature_selected_list = str(tuple1[1])
                hidden_size_list = str(tuple1[2])
                average_f1_list = str(tuple1[3])
                iteration_loss_list = str(tuple1[4])
                f.write('----------------------------------------------------\n')
                f.write('feature_switch: {}\n'.format(feature_switch_list))
                f.write('feature_selected: {}\n'.format(feature_selected_list))
                f.write('hidden_size: {}\n'.format(hidden_size_list))
                f.write('average_f1: {}\n'.format(average_f1_list))
                f.write('iteration_loss: {}\n\n'.format(iteration_loss_list))

        print("save feature and topology test result complete!!!!")

    #   ====================================================================================================================
    #   regressor
    #   ====================================================================================================================

    def set_regressor(self, hidden_layer_sizes, tol=1e-8, learning_rate_init=0.001):
        self.hidden_size_list.append(hidden_layer_sizes)
        self.mlp_hidden_layer_sizes_list.append(hidden_layer_sizes)
        # self.mlp_clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
        #                              tol = tol, learning_rate_init = learning_rate_init, verbose = True,
        #                              solver = 'sgd', momentum = 0.3,  max_iter = 10000)
        self.mlp_regressor = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                                          tol=tol, learning_rate_init=learning_rate_init,
                                          max_iter=1000, random_state=1)

    def _r_feed_data(self, folder, data_per, feature_switch_tuple=None, is_random = False):
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
                    features_list = self._feature_degradation(features_list, feature_switch_tuple)
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
            random_seed = 1
            random.seed(random_seed)
            random.shuffle(combind_list)
            samples_feature_list, samples_value_list, date_str_list, stock_id_list = zip(*combind_list)
            print ("Data set shuffling complete! Random Seed: {}".format(random_seed))
        #

        return samples_feature_list, samples_value_list, date_str_list, stock_id_list

    def r_feed_and_seperate_data(self, folder, dev_per=0.2, data_per=1.0, feature_switch_tuple=None):
        print("start feeding data......")
        if feature_switch_tuple:
            print("feature_switch_tuple: ", feature_switch_tuple)

        # clear training_set, dev_set
        self.r_training_set = []
        self.r_training_value_set = []
        self.r_dev_set = []
        self.r_dev_value_set = []

        # cut the number of training sample
        samples_feature_list, samples_value_list, date_str_list, stock_id_list = self._r_feed_data(folder, data_per,
                                                                                                   feature_switch_tuple=feature_switch_tuple)

        sample_number = len(samples_feature_list)
        dev_sample_num = math.floor(sample_number * dev_per) * -1

        self.r_training_set = samples_feature_list[0:dev_sample_num]
        self.r_training_value_set = samples_value_list[0:dev_sample_num]
        self.r_dev_set = samples_feature_list[dev_sample_num:]
        self.r_dev_value_set = samples_value_list[dev_sample_num:]
        self.r_dev_date_set = date_str_list[dev_sample_num:]
        self.r_dev_stock_id_set = stock_id_list[dev_sample_num:]

        print("r_training_set_size: {}, r_dev_set_size: {}".format(len(self.r_training_set), len(self.r_dev_set)))

    def regressor_train(self, save_clsfy_path="mlp_regressor", is_cv = False):
        self.mlp_regressor.fit(self.r_training_set, self.r_training_value_set)
        self.iteration_loss_list.append((self.mlp_regressor.n_iter_, self.mlp_regressor.loss_))
        pickle.dump(self.mlp_regressor, open(save_clsfy_path, "wb"))

        # # <debug_print>
        # if is_cv:
        #     print ("Training complete! Training Set size: {}".format(len(self.r_training_value_set)))
        # # <debug_print>

    def regressor_dev(self, save_clsfy_path="mlp_regressor", is_cv = False):
        #print("get regressor from {}.".format(save_clsfy_path))
        mlp_regressor = pickle.load(open(save_clsfy_path, "rb"))
        pred_value_list = np.array(mlp_regressor.predict(self.r_dev_set))
        actual_value_list = np.array(self.r_dev_value_set)
        mrse = calculate_mrse(actual_value_list, pred_value_list)
        date_list = self.r_dev_date_set
        stock_id_list = self.r_dev_stock_id_set
        avg_price_change = self._get_avg_price_change(pred_value_list, actual_value_list, date_list, stock_id_list)

        # count how many predicted value has the same polarity as actual value
        polar_list = [1 for x, y in zip(pred_value_list, actual_value_list) if x * y >= 0]
        polar_count = len(polar_list)
        polar_percent = polar_count / len(pred_value_list)
        #

        self.mres_list.append(mrse)
        self.avg_price_change_list.append(avg_price_change)
        self.avg_price_change_list.append(avg_price_change)
        self.polar_accuracy_list.append(polar_percent)

        # <uncomment for debugging>
        # if not is_cv:
        #     print("----------------------------------------------------------------------------------------")
        #     print("actual_value_list, ", actual_value_list)
        #     print("pred_value_list, ", pred_value_list)
        #     print("polarity: {}".format(polar_percent))
        #     print("mrse: {}".format(mrse))
        #     print("avg_price_change: {}".format(avg_price_change))
        #     print("----------------------------------------------------------------------------------------")
        # else:
        #     print("Testing complete! Testing Set size: {}".format(len(self.r_dev_value_set)))
        # <uncomment for debugging>


    def _get_avg_price_change(self, pred_value_list, actual_value_list, date_list, stock_id_list):

        # construct stock_pred_v_dict
        stock_pred_v_dict = collections.defaultdict(lambda: [])
        for i, date in enumerate(date_list):
            stock_pred_v_pair = (stock_id_list[i], pred_value_list[i])
            stock_pred_v_dict[date].append(stock_pred_v_pair)
        #

        #
        stock_actual_v_dict = collections.defaultdict(lambda: 0)
        for i, date in enumerate(date_list):
            date_stock_id_pair = (date, stock_id_list[i])
            stock_actual_v_dict[date_stock_id_pair] = actual_value_list[i]
        #


        # find the stock with the highest predicted priceChange and compute the avg priceChange
        actual_price_change_sum = 0

        for date, stock_pred_v_pair_list in stock_pred_v_dict.items():
            sorted_stock_pred_v_pair_list = sorted(stock_pred_v_pair_list, key=lambda x: x[1], reverse=True)
            best_stock_id = sorted_stock_pred_v_pair_list[0][0]
            best_stock_pred_price_change = sorted_stock_pred_v_pair_list[0][1]

            date_stock_id_pair = (date, best_stock_id)

            actual_price_change = stock_actual_v_dict[date_stock_id_pair]
            actual_price_change_sum += actual_price_change

            # # temp print
            # print ("best_stock_pred_price_change: ", best_stock_pred_price_change)
            # print("date_stock_id_pair: ", date_stock_id_pair)
            # print("actual_price_change: ", actual_price_change)
            # #

        #


        # compute the average
        avg_price_change = actual_price_change_sum / len(stock_pred_v_dict.keys())
        #

        return avg_price_change

    def r_topology_test(self, other_config_dict, hidden_layer_config_tuple):
        def _build_hidden_layer_sizes_list(hidden_layer_config_tuple):
            hidden_layer_node_min, hidden_layer_node_max, hidden_layer_node_step, hidden_layer_depth_min, \
            hidden_layer_depth_max = hidden_layer_config_tuple

            hidden_layer_unit_list = [x for x in range(hidden_layer_node_min, hidden_layer_node_max + 1)]
            hidden_layer_unit_list = hidden_layer_unit_list[::hidden_layer_node_step]
            #

            hidden_layer_layer_list = [x for x in range(hidden_layer_depth_min, hidden_layer_depth_max + 1)]
            #
            hidden_layer_sizes_list = list(itertools.product(hidden_layer_unit_list, hidden_layer_layer_list))
            return hidden_layer_sizes_list

        # :::topology_test:::




        hidden_layer_sizes_list = _build_hidden_layer_sizes_list(hidden_layer_config_tuple)
        hidden_layer_sizes_combination = len(hidden_layer_sizes_list)
        print("Total {} hidden layer size combination to test".format(hidden_layer_sizes_combination))

        learning_rate_init = other_config_dict['learning_rate_init']
        clf_path = other_config_dict['clf_path']
        tol = other_config_dict['tol']

        for i, hidden_layer_sizes in enumerate(hidden_layer_sizes_list):
            # _update_feature_switch_list
            self._update_feature_switch_list(i)

            self.set_regressor(hidden_layer_sizes, learning_rate_init=learning_rate_init, tol=tol)
            self.regressor_train(save_clsfy_path=clf_path)
            self.regressor_dev(save_clsfy_path=clf_path)
            print("==================================")
            print("Completeness: {:.5f}".format((i + 1) / hidden_layer_sizes_combination))
            print("==================================")

    def r_save_feature_topology_result(self, path, key='mres'):

        if key == 'mres':
            topology_list = sorted(list(zip(self.feature_switch_list, self.feature_selected_list,
                                            self.hidden_size_list, self.iteration_loss_list, self.polar_accuracy_list,
                                            self.avg_price_change_list, self.mres_list)),
                                   key=lambda x: x[-1])
        elif key == 'avg_pc':
            topology_list = sorted(list(zip(self.feature_switch_list, self.feature_selected_list,
                                            self.hidden_size_list, self.iteration_loss_list, self.polar_accuracy_list,
                                            self.avg_price_change_list, self.mres_list)),
                                   key=lambda x: x[-2], reverse=True)
        else:
            print("Key should be mres or avg_pc, key: {}".format(key))

        with open(path, 'w', encoding='utf-8') as f:
            for tuple1 in topology_list:
                feature_switch = str(tuple1[0])
                feature_selected = str(tuple1[1])
                hidden_size = str(tuple1[2])
                iteration_loss = str(tuple1[3])
                polar_accuracy = str(tuple1[4])
                avg_price_change = str(tuple1[5])
                mres = str(tuple1[6])
                f.write('----------------------------------------------------\n')
                f.write('feature_switch: {}\n'.format(feature_switch))
                f.write('feature_selected: {}\n'.format(feature_selected))
                f.write('hidden_size: {}\n'.format(hidden_size))
                f.write('iteration_loss: {}\n'.format(iteration_loss))
                f.write('polar_accuracy: {}\n'.format(polar_accuracy))
                f.write('avg_price_change: {}\n'.format(avg_price_change))
                f.write('mres: {}\n'.format(mres))

        print("save topology test result by {} to {} sucessfully".format(key, path))

    def get_full_feature_switch_tuple(self, folder):
        # read feature length
        file_name_list = os.listdir(folder)
        file_path_0 = [os.path.join(folder, x) for x in file_name_list][0]
        with open(file_path_0, 'r', encoding='utf-8') as f:
            feature_name_list = f.readlines()[0].split(',')[::2]
        full_feature_switch_tuple = tuple([1 for x in feature_name_list])
        return full_feature_switch_tuple

    #   ====================================================================================================================
    #   regressor END
    #   ====================================================================================================================



    #   ====================================================================================================================
    #   CROSS VALIDATION FUNCTIONS FOR REGRESSION
    #   ====================================================================================================================

    def cv_r_feed_data_train_test(self, validation_index, samples_feature_list, samples_value_list,
                                  date_str_list, stock_id_list, is_random = False, feature_switch_tuple=None,
                                  is_print = False):
        '''10 cross validation split data'''
        data_per = 1.0
        dev_per = 0.1


        if feature_switch_tuple:
            print("feature_switch_tuple: ", feature_switch_tuple)

        # clear training_set, dev_set
        self.r_training_set = []
        self.r_training_value_set = []
        self.r_dev_set = []
        self.r_dev_value_set = []


        sample_number = len(samples_feature_list)
        sample_last_index = sample_number - 1
        dev_sample_num = math.floor(sample_number * dev_per)
        validation_split_list = [[i*dev_sample_num, (i+1)*dev_sample_num] for i in range(10)]
        validation_split_list[-1][1] = sample_last_index

        # print ("validation_split_list: ", validation_split_list)
        dev_start_index = validation_split_list[validation_index][0]
        dev_end_index = validation_split_list[validation_index][1]

        self.r_training_set = samples_feature_list[0:dev_start_index] + samples_feature_list[dev_end_index:sample_last_index]
        self.r_training_value_set = samples_value_list[0:dev_start_index] + samples_value_list[dev_end_index:sample_last_index]
        self.r_dev_set = samples_feature_list[dev_start_index:dev_end_index]
        self.r_dev_value_set = samples_value_list[dev_start_index:dev_end_index]
        self.r_dev_date_set = date_str_list[dev_start_index:dev_end_index]
        self.r_dev_stock_id_set = stock_id_list[dev_start_index:dev_end_index]

        if is_print:
            print ("-------------------------------------------------------------------------")
            print ("Set data for validation index: {}, range: ({}, {})".format(validation_index,
                                                                                     dev_start_index, dev_end_index))
            print("-------------------------------------------------------------------------")




    def cv_r_topology_test(self, input_folder, feature_switch_tuple, other_config_dict,
                           hidden_layer_config_tuple, is_random = False):
        '''10 cross validation test for mlp regressor'''
        def _build_hidden_layer_sizes_list(hidden_layer_config_tuple):
            hidden_layer_node_min, hidden_layer_node_max, hidden_layer_node_step, hidden_layer_depth_min, \
            hidden_layer_depth_max = hidden_layer_config_tuple

            hidden_layer_unit_list = [x for x in range(hidden_layer_node_min, hidden_layer_node_max + 1)]
            hidden_layer_unit_list = hidden_layer_unit_list[::hidden_layer_node_step]
            #

            hidden_layer_layer_list = [x for x in range(hidden_layer_depth_min, hidden_layer_depth_max + 1)]
            #
            hidden_layer_sizes_list = list(itertools.product(hidden_layer_unit_list, hidden_layer_layer_list))
            return hidden_layer_sizes_list

        # :::topology_test:::


        # ==============================================================================================================
        # Cross Validation Train And Test
        # ==============================================================================================================

        # (1.) read the whole data set
        # cut the number of training sample
        samples_feature_list, samples_value_list, \
        date_str_list, stock_id_list = self._r_feed_data(input_folder, data_per = 1.0,
                                                         feature_switch_tuple=feature_switch_tuple, is_random = True)
        # --------------------------------------------------------------------------------------------------------------

        # (2.) construct hidden layer size list
        hidden_layer_sizes_list = _build_hidden_layer_sizes_list(hidden_layer_config_tuple)
        hidden_layer_sizes_combination = len(hidden_layer_sizes_list)
        print ("Total {} hidden layer size combination to test".format(hidden_layer_sizes_combination))
        # --------------------------------------------------------------------------------------------------------------

        # (3.) set MLP parameters
        learning_rate_init = other_config_dict['learning_rate_init']
        clf_path = other_config_dict['clf_path']
        tol = other_config_dict['tol']
        # --------------------------------------------------------------------------------------------------------------

        # (4.) test the performance of different topology of MLP by 10-cross validation
        for i, hidden_layer_sizes in enumerate(hidden_layer_sizes_list):
            print ("====================================================================")
            print ("Topology: {} starts training and testing".format(hidden_layer_sizes))
            print ("====================================================================")


            self._update_feature_switch_list(i)

            # (a.) clear the evaluation list for one hidden layer topology
            self.iteration_loss_list = []
            self.mres_list = []
            self.avg_price_change_list = []
            self.polar_accuracy_list = []
            #

            # (b.) 10-cross-validation train and test
            self.set_regressor(hidden_layer_sizes, learning_rate_init=learning_rate_init, tol=tol)
            for validation_index in range(10):
                self.cv_r_feed_data_train_test(validation_index, samples_feature_list, samples_value_list,
                                      date_str_list, stock_id_list,)
                self.regressor_train(save_clsfy_path=clf_path, is_cv = True)
                self.regressor_dev(save_clsfy_path=clf_path, is_cv = True)
            #

            # (c.) save the 10-cross-valiation evaluate result for each topology
            self.cv_iteration_loss_list.append(self.iteration_loss_list)
            self.cv_mres_list.append(self.mres_list)
            self.cv_avg_price_change_list.append(self.avg_price_change_list)
            self.cv_polar_accuracy_list.append(self.polar_accuracy_list)

            # (d.) real-time print
            print ("====================================================================")
            print ("Average mres: {}".format(np.average(self.mres_list)))
            print ("Average price change: {}".format(np.average(self.avg_price_change_list)))
            print ("Average polarity: {}".format(np.average(self.polar_accuracy_list)))
            print ("Average iteration_loss: {}".format(np.average(np.average([x[1] for x in self.iteration_loss_list]))))
            print ("====================================================================")
            print ("Completeness: {:.5f}".format((i+1)/hidden_layer_sizes_combination))
            print ("====================================================================")
            #
        # ==============================================================================================================
        # Cross Validation Train And Test END
        # ==============================================================================================================

        # --------------------------------------------------------------------------------------------------------------


    def cv_r_save_feature_topology_result(self, path, key='mres'):

        # compute the average for each list
        self.cv_iteration_loss_list = [np.average(x) for x in self.cv_iteration_loss_list]
        self.cv_polar_accuracy_list = [np.average(x) for x in self.cv_polar_accuracy_list]
        self.cv_avg_price_change_list = [np.average(x) for x in self.cv_avg_price_change_list]
        self.cv_mres_list = [np.average(x) for x in self.cv_mres_list]

        if key == 'mres':
            topology_list = sorted(list(zip(self.feature_switch_list, self.feature_selected_list,
                                            self.hidden_size_list, self.cv_iteration_loss_list,
                                            self.cv_polar_accuracy_list,
                                            self.cv_avg_price_change_list, self.cv_mres_list)),
                                   key=lambda x: x[-1])
        elif key == 'avg_pc':
            topology_list = sorted(list(zip(self.feature_switch_list, self.feature_selected_list,
                                            self.hidden_size_list, self.cv_iteration_loss_list,
                                            self.cv_polar_accuracy_list,
                                            self.cv_avg_price_change_list, self.cv_mres_list)),
                                   key=lambda x: x[-2], reverse = True)

        elif key == 'polar':
            topology_list = sorted(list(zip(self.feature_switch_list, self.feature_selected_list,
                                        self.hidden_size_list, self.cv_iteration_loss_list,
                                        self.cv_polar_accuracy_list,
                                        self.cv_avg_price_change_list, self.cv_mres_list)),
                                key=lambda x: x[-3], reverse=True)
        else:
            print("Key should be mres or avg_pc, key: {}".format(key))

        with open(path, 'w', encoding='utf-8') as f:
            for tuple1 in topology_list:
                feature_switch = str(tuple1[0])
                feature_selected = str(tuple1[1])
                hidden_size = str(tuple1[2])
                iteration_loss = str(tuple1[3])
                polar_accuracy = str(tuple1[4])
                avg_price_change = str(tuple1[5])
                mres = str(tuple1[6])
                f.write('----------------------------------------------------\n')
                f.write('feature_switch: {}\n'.format(feature_switch))
                f.write('feature_selected: {}\n'.format(feature_selected))
                f.write('hidden_size: {}\n'.format(hidden_size))
                f.write('average_iteration_loss: {}\n'.format(iteration_loss))
                f.write('average_polar_accuracy: {}\n'.format(polar_accuracy))
                f.write('average_avg_price_change: {}\n'.format(avg_price_change))
                f.write('average_mres: {}\n'.format(mres))

        print("Regression! Save 10-cross-validation topology test result by {} to {} sucessfully!".format(key, path))


        # ==============================================================================================================

    #   ====================================================================================================================
    #   CROSS VALIDATION FUNCTIONS FOR REGRESSION END
    #   ====================================================================================================================


    #   ====================================================================================================================
    #   CROSS VALIDATION FUNCTIONS FOR CLASSIFICATION
    #   ====================================================================================================================
    def cv_feed_and_seperate_data(self, validation_index, samples_feature_list, samples_label_list,
                                  dev_per=0.1, is_print = False):


        # clear training_set, dev_set
        self.training_set = []
        self.training_label = []
        self.dev_set = []
        self.dev_label = []
        #


        sample_number = len(samples_feature_list)
        sample_last_index = sample_number - 1
        dev_sample_num = math.floor(sample_number * dev_per)
        validation_split_list = [[i*dev_sample_num, (i+1)*dev_sample_num] for i in range(10)]
        validation_split_list[-1][1] = sample_last_index

        # print ("validation_split_list: ", validation_split_list)
        dev_start_index = validation_split_list[validation_index][0]
        dev_end_index = validation_split_list[validation_index][1]


        self.training_set = samples_feature_list[0:dev_start_index] + samples_feature_list[dev_end_index:sample_last_index]
        self.training_label = samples_label_list[0:dev_start_index] + samples_label_list[dev_end_index:sample_last_index]
        self.dev_set = samples_feature_list[dev_start_index:dev_end_index]
        self.dev_label = samples_label_list[dev_start_index:dev_end_index]

        # count the label in traning and testing data
        dev_label_dict = collections.defaultdict(lambda: 0)
        for label in self.dev_label:
            dev_label_dict[label] += 1

        training_label_dict = collections.defaultdict(lambda: 0)
        for label in self.training_label:
            training_label_dict[label] += 1

        if is_print:
            print("-------------------------------------------------------------------------\n")
            print ("-------------------------------------------------------------------------")
            print ("Set data for validation index: {}, range: ({}, {})".format(validation_index,
                                                                                     dev_start_index, dev_end_index))
            #print ("Training Label: {}".format(training_label_dict.items()))
            print ("Dev Label: {}".format(dict(dev_label_dict.items())))
            print ("-------------------------------------------------------------------------")





    def cv_cls_topology_test(self, input_folder, feature_switch_tuple, other_config_dict,
                           hidden_layer_config_tuple, is_random=False):
        '''10 cross validation test for mlp classifier'''

        def _build_hidden_layer_sizes_list(hidden_layer_config_tuple):
            hidden_layer_node_min, hidden_layer_node_max, hidden_layer_node_step, hidden_layer_depth_min, \
            hidden_layer_depth_max = hidden_layer_config_tuple

            hidden_layer_unit_list = [x for x in range(hidden_layer_node_min, hidden_layer_node_max + 1)]
            hidden_layer_unit_list = hidden_layer_unit_list[::hidden_layer_node_step]
            #

            hidden_layer_layer_list = [x for x in range(hidden_layer_depth_min, hidden_layer_depth_max + 1)]
            #
            hidden_layer_sizes_list = list(itertools.product(hidden_layer_unit_list, hidden_layer_layer_list))
            return hidden_layer_sizes_list

        # :::topology_test:::


        # ==============================================================================================================
        # Cross Validation Train And Test
        # ==============================================================================================================

        # feature switch tuple
        if feature_switch_tuple:
            self.feature_switch_list.append(feature_switch_tuple)

        # (1.) read the whole data set
        # cut the number of training sample
        data_per = 1.0
        samples_feature_list, samples_label_list = self._feed_data(input_folder, data_per, is_random = is_random)
        # --------------------------------------------------------------------------------------------------------------

        # (2.) construct hidden layer size list
        hidden_layer_sizes_list = _build_hidden_layer_sizes_list(hidden_layer_config_tuple)
        hidden_layer_sizes_combination = len(hidden_layer_sizes_list)
        print("Total {} hidden layer size combination to test".format(hidden_layer_sizes_combination))
        # --------------------------------------------------------------------------------------------------------------

        # (3.) set MLP parameters
        learning_rate_init = other_config_dict['learning_rate_init']
        clf_path = other_config_dict['clf_path']
        tol = other_config_dict['tol']
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

            # (b.) 10-cross-validation train and test
            self.set_mlp(hidden_layer_sizes, learning_rate_init=learning_rate_init, tol=tol)
            for validation_index in range(10):
                self.cv_feed_and_seperate_data(validation_index, samples_feature_list, samples_label_list,
                                               dev_per=0.1, is_print = False)
                self.train(save_clsfy_path=clf_path, is_cv=True)
                self.dev(save_clsfy_path=clf_path, is_cv=True)
            #

            # (c.) save the 10-cross-valiation evaluate result for each topology

            self.cv_iteration_loss_list.append([x[1] for x in self.iteration_loss_list])
            self.cv_average_average_f1_list.append(self.average_f1_list)
            self.cv_average_accuracy_list.append(self.accuracy_list)

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

    def cv_cls_save_feature_topology_result(self, path):

        # compute the average for each list
        self.cv_iteration_loss_list = [np.average(x) for x in self.cv_iteration_loss_list]
        self.cv_average_accuracy_list = [np.average(x) for x in self.cv_average_accuracy_list]
        self.cv_average_average_f1_list = [np.average(x) for x in self.cv_average_average_f1_list]

        topology_list = sorted(list(zip(self.feature_switch_list, self.feature_selected_list,
                                            self.hidden_size_list, self.cv_iteration_loss_list,
                                            self.cv_average_accuracy_list,
                                            self.cv_average_average_f1_list)),
                                   key=lambda x: x[-1], reverse=True)


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