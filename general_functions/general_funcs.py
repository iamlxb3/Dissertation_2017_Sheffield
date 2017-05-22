import datetime
import math

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

    print (stop_index_list)
    for stop_index_tuple in stop_index_list:
        split_list.append(list1[stop_index_tuple[0]:stop_index_tuple[1]])

    return split_list