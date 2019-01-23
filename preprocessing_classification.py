import numpy as np
import pandas as pd
import datetime

'''
Takes data and combines U, V, W for each weight and load. 
Each line of the resulting CSV contains k samples of each type (u,v,w) signals.
Its shape  is: (KxU), (kxV), (kxW), load, rpm

K (int): number samples per line
path (String): path to the folder containing the data (All files need to be in the folder, without any subfolders)
'''
def split_data(k, path):
    rpms = ['375', '750', '1500']
    loads = ['FL' , 'HL', 'NL']

    start_time = datetime.datetime.now()

    for rpm in rpms:
        for load in loads:
            print ("Transforming " + rpm + " " + load)

            print ("\tReading data")
            data_u = pd.read_csv(path + '/H' + rpm + load +'U.txt')
            data_v = pd.read_csv(path + '/H' + rpm + load +'V.txt')
            data_w = pd.read_csv(path + '/H' + rpm + load +'W.txt')

            data = pd.concat([data_u, data_v, data_w], axis=1)
           
            print("\tSplitting")
            data_split = np.split(data, len(data)/k)
            
            print("\tWriting to csv")
            to_csv(data_split, load, rpm, k)
    
    # print("Total time elapsed: " + str(datetime.datetime.now() - start_time))


'''
Writes already split data into CSV
'''
def to_csv(data_split, load, rpm, k):
    file = open("data" + rpm + load + str(k) + ".txt", "w+")
    
    #CSV header
    for column in data_split[0]:
        for i in range(0,len(data_split[0])):
            file.write(str(column) + ", ")
    file.write("load, rpms\n")

    # CSV body
    for split in data_split:
        for column in split:
            for elem in split[column]:
                file.write(str(elem) + ", ")
                
        file.write(load + ", " + rpm +  "\n")


'''
Reads already split data and build a file with p samples in total with a uniform dist.

p (int): number of samples in final file
path (String): path to a,ready slpit data
k (int): number of samples of each type in the split data
'''
def sample_data(p, path, k):
    rpms = ['375', '750', '1500']
    loads = ['FL' , 'HL', 'NL']

    data_sample = pd.DataFrame()
    for rpm in rpms:
        for load in loads:
            data_case = pd.read_csv(path + 'data' + rpm + load + str(k) + '.txt')
            data_case_sample = data_case.sample(int(p/9))
            data_sample = data_sample.append(data_case_sample)

    data_sample = data_sample.sample(frac=1)
    data_sample.to_csv('sample' + str(p) + 'x' + str(k) + '.txt', index=False)


k = [128, 64, 32, 16, 8, 4]

 for k in k:
     split_data(k, 'Data')
     sample(7000, '', k)
     print('Done ' + str(k))