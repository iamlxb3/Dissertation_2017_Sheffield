import numpy as np
import math
import matplotlib.pyplot as plt
import collections

capital = 10000
avg_profit = 0.136

mu = avg_profit
sigma = 0.23 # mean and standard deviation
week_num = 52

profit_range_count_list = []
week_profit_list = np.random.normal(mu, sigma, week_num)
profit_range_list = np.arange(-1.1, 1.1, 0.01)
print ("profit_range_list: ", profit_range_list)


profit_range_tuple_list  = [(x, profit_range_list[i+1]) for i, x in enumerate(profit_range_list) if i <= len(profit_range_list) - 2]
for profit_range_tuple in profit_range_tuple_list:
    low_limit = profit_range_tuple[0]
    up_limit = profit_range_tuple[1]
    range_count = ((week_profit_list >= low_limit) & (week_profit_list < up_limit)).sum()
    profit_range_count_list.append(range_count)



print ("week_profit_list: ", week_profit_list)
week_list = [x for x in range(week_num)]

# plot
plt.bar([x for x in range(len(profit_range_count_list))], profit_range_count_list, align='center')
plt.show()
#


for week_profit in week_profit_list:
    capital = capital*(1+week_profit)
    
print ("capital: ", capital)