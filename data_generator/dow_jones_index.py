import sys
import os
import collections
import re
import datetime
import time



class DowJonesIndex:
    """
    quarter:  the yearly quarter (1 = Jan-Mar; 2 = Apr=Jun).
    stock: the stock symbol (see above)
    date: the last business day of the work (this is typically a Friday)
    open: the price of the stock at the beginning of the week
    high: the highest price of the stock during the week
    low: the lowest price of the stock during the week
    close: the price of the stock at the end of the week
    volume: the number of shares of stock that traded hands in the week
    percent_change_price: the percentage change in price throughout the week
    percent_chagne_volume_over_last_wek: the percentage change in the number of shares of
    stock that traded hands for this week compared to the previous week
    previous_weeks_volume: the number of shares of stock that traded hands in the previous week
    next_weeks_open: the opening price of the stock in the following week
    next_weeks_close: the closing price of the stock in the following week
    percent_change_next_weeks_price: the percentage change in price of the stock in the following week
    days_to_next_dividend: the number of days until the next dividend
    percent_return_next_dividend: the percentage of return on the next dividend
    """
    def __init__(self):
        pass

    def format_raw_data(self, input_file, save_folder):
        with open(input_file, 'r', encoding = 'utf-8') as f:
            for i, line in enumerate(f):
                if i == 0:
                    feature_name_list = line.split(',')
                    continue
                line_list = line.split(',')

                # get the file name
                stock_name = line_list[1]
                quarter = line_list[0]
                date = line_list[2]
                date_temp = time.strptime(date, '%m/%d/%Y')
                date_obj = datetime.datetime(*date_temp[:3])
                date_str = date_obj.strftime("%Y-%m-%d")
                file_name = date_str + '_' + quarter + '_' + stock_name + '.txt'
                #

                # get the feature value list
                feature_value_list = line_list[3:]

                # mark nan data
                feature_value_list =  ['nan' if not x else x for x in feature_value_list]

                # get rid of $
                feature_value_list = [re.findall(r'[0-9\.]+', x)[0] if x != 'nan' else x for x in feature_value_list ]
                feature_name_value_list = [j for i in zip(feature_name_list[3:], feature_value_list) for j in i]
                feature_name_value_str = ",".join(feature_name_value_list)

                # save the file
                file_path = os.path.join(save_folder, file_name)
                with open(file_path, 'w', encoding = 'utf-8') as f:
                    f.write(feature_name_value_str)


