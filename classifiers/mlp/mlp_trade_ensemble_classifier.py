# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# MLP classifier only for trading, currently for a share stock and forex
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# (c) 2017 PJS, University of Sheffield, iamlxb3@gmail.com
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# ==========================================================================================================
# general import
# ==========================================================================================================
import pickle
import collections
import numpy as np
import sys
import math
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
from mlp_trade_classifier import MlpTradeClassifier
from trade_general_funcs import compute_average_f1
# ==========================================================================================================




# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class MlpTradeDataEnsembleClassifier(MlpTradeClassifier, ):
    '''Ensemble classifier of different data'''

    def __init__(self, ensemble_number):
        super().__init__()
        self.ensemble_number = ensemble_number

    def set_mlp_clf(self, hidden_layer_sizes, tol=1e-6, learning_rate_init=0.001, verbose=False):

        self.hidden_size_list.append(hidden_layer_sizes)
        self.mlp_hidden_layer_sizes_list.append(hidden_layer_sizes)

        self.ensemble_clf_list = []
        for i in range(self.ensemble_number):
            temp_clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                         tol=tol, learning_rate_init=learning_rate_init,
                                         max_iter=2000, random_state=1, verbose=verbose)
            self.ensemble_clf_list.append(temp_clf)



    def clf_train(self, save_clsfy_path="mlp_trade_classifier", is_production=False):

        # split data and train
        training_set_size = len(self.training_set)
        block_size = math.floor(training_set_size/self.ensemble_number)
        n_iter_list = []
        loss_list = []

        for i in range(self.ensemble_number):
            if i == self.ensemble_number - 1:
                training_set = self.training_set[i * block_size:training_set_size]
                training_value_set = self.training_value_set[i * block_size:training_set_size]
            else:
                training_set = self.training_set[i*block_size:i*block_size + block_size]
                training_value_set = self.training_value_set[i*block_size:i*block_size + block_size]

            self.ensemble_clf_list[i].fit(training_set, training_value_set)
            n_iter_list.append(self.ensemble_clf_list[i].n_iter_)
            loss_list.append(self.ensemble_clf_list[i].loss_)


        self.iteration_loss_list.append((np.average(n_iter_list), np.average(loss_list)))


        for i, classfier in enumerate(self.ensemble_clf_list):
            save_clsfy_path = save_clsfy_path + '_data_ensemble_{}'.format(i)
            pickle.dump(classfier, open(save_clsfy_path, "wb"))



    def clf_dev(self, save_clsfy_path="mlp_trade_classifier", is_cv=False, is_return = False):

        ensemble_classifier_list = []
        for i in range(self.ensemble_number):
            save_clsfy_path = save_clsfy_path + '_data_ensemble_{}'.format(i)
            classfier = pickle.load(open(save_clsfy_path, "rb"))
            ensemble_classifier_list.append(classfier)

        pred_label_list_ensemble = []
        for i, classfier in enumerate(ensemble_classifier_list):
            pred_label_list_1 = classfier.predict(self.dev_set)
            pred_label_list_ensemble.append(pred_label_list_1)

        # find the label with the most vote
        pred_label_list = []
        pred_label_list_ensemble = list(zip(*pred_label_list_ensemble))


        for i, label_tuple in enumerate(pred_label_list_ensemble):
            label_list = list(set(label_tuple))
            label_max_count = 0
            label_max_index = 0
            for j, label in enumerate(label_list):
                count = label_tuple.count(label)
                if count > label_max_count:
                    label_max_count = count
                    label_max_index = j

            pred_label_list.append(label_list[label_max_index])

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

        if is_return:
            return average_f1, accuracy

