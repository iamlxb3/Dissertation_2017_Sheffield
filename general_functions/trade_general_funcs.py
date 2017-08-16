import datetime
import math
import numpy as np
import random
import os
import collections
import re
import itertools
from sklearn.metrics import mean_squared_error
import sys


# ==========================================================================================================
# ADD SYS PATH
# ==========================================================================================================
parent_folder = (os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
path1 = os.path.join(parent_folder, 'strategy')
sys.path.append(path1)
# ==========================================================================================================


# ==========================================================================================================
# local package import
# ==========================================================================================================
from a_share_strategy import top_n_avg_strategy
# ==========================================================================================================




def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)


def split_list_by_percentage(per_tuple, list1):
    list_len = len(list1)
    split_list = []

    stop_index_list = []
    for i, per in enumerate(per_tuple):

        stop_index = math.ceil(per * list_len)
        if i == 0:
            stop_index_tuple = (0, stop_index)
        elif i == len(per_tuple) - 1:
            stop_index_tuple = (previous_stop_index, len(list1))
        else:
            stop_index_tuple = (previous_stop_index, stop_index)

        stop_index_list.append(stop_index_tuple)
        previous_stop_index = stop_index

    # print (stop_index_list)
    for stop_index_tuple in stop_index_list:
        split_list.append(list1[stop_index_tuple[0]:stop_index_tuple[1]])

    return split_list


def calculate_mrse(actual_value_array, pred_value_array):
    '''root-mean-square error sk learn'''
    # TODO try and except for debug
    try:
        rmse = math.sqrt(mean_squared_error(actual_value_array, pred_value_array))
    except ValueError:
        print ("actual_value_array: ", actual_value_array)
        print ("pred_value_array: ", pred_value_array)
        sys.exit()
    return rmse


def calculate_mrse_PJS(golden_value_array, pred_value_array):
    '''root-mean-square error'''
    if len(golden_value_array) != len(pred_value_array):
        print("golden_value_array len is not equal to pred_value_array len {}".
              format(golden_value_array, pred_value_array))
        return None
    sample_count = len(golden_value_array)
    rmse = golden_value_array - pred_value_array
    rmse = rmse ** 2
    rmse = np.sum(rmse)
    rmse = rmse / sample_count
    rmse = np.sqrt(rmse)
    return rmse


def list_by_index(list1, index_list):
    new_list = [list1[index] for index in index_list]
    return new_list


def create_random_sub_set_list(set1, sub_set_size, random_seed=1):
    sub_set_list = []
    while (len(set1)) >= sub_set_size:
        set1_list = sorted(list(set1))
        random.seed(random_seed)
        sub_set = set(random.sample(set1_list, sub_set_size))
        set1 -= sub_set
        sub_set_list.append(sub_set)
    return sub_set_list


def count_label(folder):
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


def feature_degradation(features_list, feature_switch_tuple):
    new_feature_list = []
    for i, switch in enumerate(feature_switch_tuple):
        if switch == 1:
            new_feature_list.append(features_list[i])
    return new_feature_list

def compute_average_f1(pred_label_list, gold_label_list):
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


def generate_feature_switch_list(folder):
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


def get_avg_price_change(pred_value_list, actual_value_list, date_list,
                          stock_id_list, include_top_list=[1]):
    avg_price_change_list = []
    var_list = []
    std_list = []
    # construct stock_pred_v_dict
    stock_pred_v_dict = collections.defaultdict(lambda: [])
    for i, date in enumerate(date_list):
        stock_pred_v_pair = (stock_id_list[i], pred_value_list[i])
        stock_pred_v_dict[date].append(stock_pred_v_pair)
    #
    stock_actual_v_dict = collections.defaultdict(lambda: 0)
    for i, date in enumerate(date_list):
        date_stock_id_pair = (date, stock_id_list[i])
        stock_actual_v_dict[date_stock_id_pair] = actual_value_list[i]
    #
    for include_top in include_top_list:
        # (0.) avg_price_change
        avg_price_change, var, std = top_n_avg_strategy(stock_actual_v_dict, stock_pred_v_dict,
                                                        include_top=include_top)
        avg_price_change_list.append(avg_price_change)
        var_list.append(var)
        std_list.append(std)

    return tuple(avg_price_change_list), tuple(var_list), tuple(std_list)


def get_full_feature_switch_tuple(folder):
    # read feature length
    file_name_list = os.listdir(folder)
    file_path_0 = [os.path.join(folder, x) for x in file_name_list][0]
    with open(file_path_0, 'r', encoding='utf-8') as f:
        feature_name_list = f.readlines()[0].split(',')[::2]
    full_feature_switch_tuple = tuple([1 for x in feature_name_list])
    return full_feature_switch_tuple


def build_hidden_layer_sizes_list(hidden_layer_config_tuple):
    hidden_layer_node_min, hidden_layer_node_max, hidden_layer_node_step, hidden_layer_depth_min, \
    hidden_layer_depth_max = hidden_layer_config_tuple

    hidden_layer_unit_list = [x for x in range(hidden_layer_node_min, hidden_layer_node_max + 1)]
    hidden_layer_unit_list = hidden_layer_unit_list[::hidden_layer_node_step]
    #

    hidden_layer_layer_list = [x for x in range(hidden_layer_depth_min, hidden_layer_depth_max + 1)]
    #
    hidden_layer_sizes_list = list(itertools.product(hidden_layer_unit_list, hidden_layer_layer_list))
    return hidden_layer_sizes_list

def read_pca_component(folder_path):
    file_name_list = os.listdir(folder_path)
    file_name1 = file_name_list[0]
    file_name1_path = os.path.join(folder_path, file_name1)
    with open (file_name1_path, 'r', encoding = 'utf-8') as f:
        feature_list = f.readlines()[0].strip().split(',')[::2]
        feature_num = len(feature_list)
    print ("Read PCA n-component {}".format(feature_num))
    return feature_num