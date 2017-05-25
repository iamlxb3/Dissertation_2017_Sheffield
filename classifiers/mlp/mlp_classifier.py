from sklearn.neural_network import MLPClassifier
import os
import numpy as np
import collections
import re
import pickle
import math
import datetime
import itertools
import sys
from pjslib.logger import logger2


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
        self.average_f1_list = []
        self.label_tp_fp_tn_dict = {}
        self.feature_switch_list = []
        self.feature_selected_list = []
        self.iteration_loss_list = []

    def count_label(self, folder):
        file_name_list = os.listdir(folder)
        label_dict = collections.defaultdict(lambda: 0)
        for file_name in file_name_list:
            try:
                label = re.findall(r'_([0-9A-Za-z]+)\.', file_name)[0]
            except IndexError:
                print ("Check folder path!")
                break
            label_dict[label] += 1
        print ("label_dict: {}".format(list(label_dict.items())))

    def set_mlp(self, hidden_layer_sizes, tol = 1e-6, learning_rate_init = 0.001):
        self.hidden_size_list.append(hidden_layer_sizes)
        self.mlp_hidden_layer_sizes_list.append(hidden_layer_sizes)
        # self.mlp_clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
        #                              tol = tol, learning_rate_init = learning_rate_init, verbose = True,
        #                              solver = 'sgd', momentum = 0.3,  max_iter = 10000)
        self.mlp_clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                     tol = tol, learning_rate_init = learning_rate_init,
                                     max_iter = 1000, random_state = 1)

    def _feature_degradation(self, features_list, feature_switch_tuple):
        new_feature_list = []
        for i, switch in enumerate(feature_switch_tuple):
            if switch == 1:
                new_feature_list.append(features_list[i])
        return new_feature_list

    def _feed_data(self, folder, data_per, feature_switch_tuple = None):
        if feature_switch_tuple:
            self.feature_switch_list.append(feature_switch_tuple)
        # ::: _feed_data :::
        # TODO test the folder exists
        file_name_list = os.listdir(folder)
        file_path_list = [os.path.join(folder, x) for x in file_name_list]
        file_total_number = len(file_name_list)
        file_used_number = math.floor(data_per*file_total_number)  # restrict the number of training sample
        file_path_list = file_path_list[0:file_used_number]
        samples_feature_list = []
        samples_label_list = []
        for f_path in file_path_list:
            f_name = os.path.basename(f_path)
            label = re.findall(r'_([A-Za-z0-9]+)\.', f_name)[0]
            with open (f_path, 'r') as f:
                features_list  = f.readlines()[0].split(',')
                features_list = features_list[1::2]
                features_list = [float(x) for x in features_list]
                if feature_switch_tuple:
                    features_list = self._feature_degradation(features_list, feature_switch_tuple)
                features_array = np.array(features_list)
                #features_array = features_array.reshape(-1,1)
                samples_feature_list.append(features_array)
                samples_label_list.append(label)
        print ("read feature list and label list for {} successful!".format(folder))
        return samples_feature_list, samples_label_list

    def read_selected_feature_list(self, folder, feature_switch_list):
        file_name_list = os.listdir(folder)
        file_path_0 = [os.path.join(folder, x) for x in file_name_list][0]
        with open (file_path_0, 'r', encoding = 'utf-8') as f:
            feature_name_list = f.readlines()[0].split(',')[::2]
            selected_feature_list = self._feature_degradation(feature_name_list, feature_switch_list)
        self.feature_selected_list.append(selected_feature_list)


    def feed_and_seperate_data(self, folder, dev_per = 0.2, data_per = 1.0, feature_switch_tuple = None):

        # clear training_set, dev_set
        self.training_set = []
        self.training_label = []
        self.dev_set = []
        self.dev_label = []
        #

        samples_dict = collections.defaultdict(lambda: [])

        # cut the number of training sample
        samples_feature_list, samples_label_list = self._feed_data(folder, data_per,
                                                                   feature_switch_tuple= feature_switch_tuple)

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


        dev_label_dict = collections.defaultdict(lambda :0)
        for label in self.dev_label:
            dev_label_dict[label] += 1

        training_label_dict = collections.defaultdict(lambda :0)
        for label in self.training_label:
            training_label_dict[label] += 1

        print ("dev_label_dict: ", list(dev_label_dict.items()))
        print ("training_label_dict: ",  list(training_label_dict.items()))

        # #
        # sample_number = math.floor(len(samples_feature_list))
        # dev_sample_num = math.floor(sample_number*dev_per)
        # print("dev_sample_num: ", dev_sample_num)
        # dev_sample_num = dev_sample_num * -1
        # self.training_set = samples_feature_list[0:dev_sample_num]
        # self.training_label = samples_label_list[0:dev_sample_num]
        # self.dev_set = samples_feature_list[dev_sample_num:]
        # self.dev_label = samples_label_list[dev_sample_num:]


    def train(self, save_clsfy_path ="mlp_classifier"):
        print ("self.training_set_size: ", len(self.training_set))
        print ("self.training_label_size: ", len(self.training_label))

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
        label_tp_fp_tn_dict = collections.defaultdict(lambda :[0, 0, 0, 0]) # tp,fp,fn,f1
        label_set = set(gold_label_list)

        for i, pred_label in enumerate(pred_label_list):
            gold_label = gold_label_list[i]
            for label in label_set:
                if pred_label == label and gold_label == label:
                    label_tp_fp_tn_dict[label][0] += 1 # true positve
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
                f1 =   (2*precision*recall)/(precision + recall)              # equal weight to precision and recall
            f1_list[3] = f1
            # reference:
            # https://www.quora.com/What-is-meant-by-F-measure-Weighted-F-Measure-and-Average-F-Measure-in-NLP-Evaluation

        return label_tp_fp_tn_dict


    def dev(self, save_clsfy_path ="mlp_classifier"):
        mlp = pickle.load(open(save_clsfy_path, "rb"))
        pred_label_list = []
        for feature_array in self.dev_set:
            feature_array = feature_array.reshape(1, -1)
            pred_label = mlp.predict(feature_array)[0]
            pred_label_list.append(pred_label)

        pred_label_dict = collections.defaultdict(lambda :0)
        for pred_label in pred_label_list:
            pred_label_dict[pred_label] += 1


        label_tp_fp_tn_dict = self._compute_average_f1(pred_label_list, self.dev_label)
        self.label_tp_fp_tn_dict = label_tp_fp_tn_dict
        label_f1_list = sorted([(key, x[3]) for key, x in label_tp_fp_tn_dict.items()])
        f1_list = [x[1] for x in label_f1_list]
        average_f1 = np.average(f1_list)
        self.average_f1_list.append(average_f1)
        # delete
        # correct = 0
        # for i, pred_label in enumerate(pred_label_list):
        #     if pred_label == self.dev_label[i]:
        #         correct += 1
        # accuracy = correct/len(self.dev_label)
        # self.accuracy_list.append(accuracy)
        # delete
        print ("\n=================================================================")
        print ("Dev set result!")
        print("=================================================================")
        print ("pred_label_dict: {}".format(list(pred_label_dict.items())))
        print ("label_f1_list: {}".format(label_f1_list))
        print ("average_f1: ", average_f1)
        print("=================================================================")


    def weekly_predict(self):
        mlp = pickle.load(open("mlp_classifier", "rb"))
        # read feature txt for check
        standard_feature = []
        with open('features.txt', 'r') as f:
            for line in f:
                standard_feature.append(line.strip())
        #

        folder = "pred_raw_data"
        nearest_friday = datetime.datetime.today().date()
        delta = datetime.timedelta(days=1)
        while nearest_friday.weekday() != 4:
            nearest_friday -= delta
        nearest_friday_str = nearest_friday.strftime("%Y-%m-%d")
        file_name_list = os.listdir(folder)
        prediction_set = []

        for file_name in file_name_list:
            is_file_valid = True
            stock_id = re.findall(r'_([0-9]+).csv', file_name)[0]
            date = re.findall(r'([0-9]+-[0-9]+-[0-9]+)_', file_name)[0]
            if date != nearest_friday_str:
                print ("{} is out dated!".format(file_name))
                continue
            file_path = os.path.join(folder, file_name)
            feature_tuple_list = []
            with open(file_path, 'r') as f:
                for line in f:
                    line_list = line.split(',')
                    feature_tuple_list.append(tuple(line_list))

            # =================================================================================================
            # check features are complete
            # =================================================================================================
            # print ("feature_tuple_list: ", feature_tuple_list)
            # print ("standard_feature: ", standard_feature)
            if len(feature_tuple_list) != len(standard_feature):
                logger2.error("{} file has more or less features!".format(file_name))
                continue

            for i, (feature_name, _) in enumerate(feature_tuple_list):
                if feature_name == standard_feature[i]:
                    pass
                else:
                    logger2.error("{} file has missing or wrong feature!".format(file_name))
                    is_file_valid = False
                    break
            if not is_file_valid:
                continue
            else:
                feature_tuple_list = [float(x[1]) for x in feature_tuple_list]
            # =================================================================================================

            # =================================================================================================
            # construct features_set and predict
            # =================================================================================================
            feature_array = np.array(feature_tuple_list).reshape(1,-1)
            pred_label = mlp.predict(feature_array)[0]
            prob = 0.0
            prediction_set.append((stock_id, pred_label, prob))
            # =================================================================================================

        # manual debug



        # sort by label
        final_prediction_set = []
        is_prediction_set_not_found = True
        print ("prediction_set_size: ", len(prediction_set))
        #
        while is_prediction_set_not_found:
            prediction_set_top = [x for x in prediction_set if x[1] == 'top']
            if not prediction_set_top:
                prediction_set_good = [x for x in prediction_set if x[1] == 'good']
            else:
                final_prediction_set = prediction_set_top
                is_prediction_set_not_found = False
                break

            if not prediction_set_good:
                prediction_set_pos = [x for x in prediction_set if x[1] == 'pos']
            else:
                final_prediction_set = prediction_set_good
                is_prediction_set_not_found = False
                break

            if prediction_set_top == [] and prediction_set_good == []:
                final_prediction_set = prediction_set_pos
                is_prediction_set_not_found = False
        #

        prediction_set = sorted(final_prediction_set, key = lambda x:x[2], reverse = True)

        # write prediction result
        with open('prediction.txt', 'w') as f:
            for prediction_tuple in prediction_set:
                f.write(str(prediction_tuple)+ '\n')

    def topology_test(self, other_config_dict, hidden_layer_config_tuple):

        def _build_hidden_layer_sizes_list(hidden_layer_config_tuple):
            hidden_layer_node_min, hidden_layer_node_max, hidden_layer_node_step, hidden_layer_depth_min,\
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

            self.set_mlp(hidden_layer_sizes, learning_rate_init=learning_rate_init, tol = tol)
            self.train(save_clsfy_path=clf_path)
            self.dev(save_clsfy_path=clf_path)

        #self.save_feature_topology_result(topology_result_path)

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
        with open (file_path_0, 'r', encoding = 'utf-8') as f:
            feature_name_list = f.readlines()[0].split(',')[::2]
        feature_num = len(feature_name_list)
        feature_switch_list_all = list(itertools.product([0,1], repeat = feature_num))
        feature_switch_list_all.remove(tuple([0 for x in range(feature_num)]))
        print ("Total feature combination: {}".format(len(feature_switch_list_all)))
        return feature_switch_list_all

    def save_feature_topology_result(self, path):
        topology_list = sorted(list(zip(self.feature_switch_list, self.feature_selected_list,
                                        self.hidden_size_list, self.average_f1_list, self.iteration_loss_list)),
                                    key = lambda x:x[3], reverse = True)
        with open (path, 'w', encoding = 'utf-8') as f:
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

        print ("save feature and topology test result complete!!!!")