import datetime
import math
import numpy as np
import random
from sklearn.metrics import mean_squared_error

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)


def split_list_by_percentage(per_tuple, list1):
    list_len = len(list1)
    split_list = []

    stop_index_list = []
    for i, per in enumerate(per_tuple):

        stop_index = math.ceil(per*list_len)
        if i == 0:
            stop_index_tuple = (0, stop_index)
        elif i == len(per_tuple) - 1:
            stop_index_tuple = (previous_stop_index, len(list1))
        else:
            stop_index_tuple = (previous_stop_index, stop_index)

        stop_index_list.append(stop_index_tuple)
        previous_stop_index = stop_index

    #print (stop_index_list)
    for stop_index_tuple in stop_index_list:
        split_list.append(list1[stop_index_tuple[0]:stop_index_tuple[1]])

    return split_list



def calculate_mrse(actual_value_array, pred_value_array):
    '''root-mean-square error sk learn'''
    rmse = math.sqrt(mean_squared_error(actual_value_array, pred_value_array))
    return rmse



def calculate_mrse_PJS(golden_value_array, pred_value_array):
    '''root-mean-square error'''
    if len(golden_value_array) != len(pred_value_array):
        print ("golden_value_array len is not equal to pred_value_array len {}".
               format(golden_value_array, pred_value_array))
        return None
    sample_count = len(golden_value_array)
    rmse = golden_value_array - pred_value_array
    rmse = rmse**2
    rmse = np.sum(rmse)
    rmse = rmse / sample_count
    rmse = np.sqrt(rmse)
    return rmse

def list_by_index(list1, index_list):
    new_list = [list1[index] for index in index_list]
    return new_list

def create_random_sub_set_list(set1, sub_set_size, random_seed = 1):

    sub_set_list = []
    while (len(set1)) >= sub_set_size:
        random.seed(random_seed)
        set1_list = sorted(list(set1))
        sub_set = set(random.sample(set1_list, sub_set_size))
        set1 -= sub_set
        sub_set_list.append(sub_set)
    return sub_set_list