import matplotlib.pyplot as plt
import numpy

with open('loss_test.csv', 'r') as f:
    file_list = f.readlines()
    file_list = [x.strip().split(',') for x in file_list]
    loss_list = [x[0] for x in file_list]
    iteration_num = [x[1] for x in file_list]
    rmse = [x[2] for x in file_list]
    avg_pc = [float(x[3]) for x in file_list]
    polar_percent = [x[4] for x in file_list]
    
    
    
    pos_neg = [1 if x > 0 else 0 for x in avg_pc ]
    one_count = pos_neg.count(1)
    zero_count = pos_neg.count(0)
    print ("one_count ", one_count)
    print ("zero_count ", zero_count)

    
plt.plot(loss_list, rmse, 'rx', label = 'rmse')
plt.legend()
plt.show()

plt.plot(loss_list, avg_pc, 'bx', label = 'avg_pc')
plt.legend()
plt.show()

plt.plot(loss_list, polar_percent, 'gx', label = 'hit')

plt.legend()
plt.show()