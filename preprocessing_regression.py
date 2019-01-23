import numpy as np
import pandas as pd

'''
Takes data and combines U, V, W for each weight and load. 
Each line of the resulting CSV contains k samples of each type (u,v,w) signals.
Its shape  is: (KxU), (kxV), (kxW), load, rpm

Here we have different loads and the file format is different

K (int): number samples per line
path (String): path to the folder containing the data (All files need to be in the folder, without any subfolders)
'''
def split_data(k, path):
    loads = ['1.5','2.5','3.5','4.5','5.5','6.5','7.5','8.5','9.5','10.5']
    for load in loads:
        print ("Transforming " + " " + load)
        print ("\tReading data")
        data_u = pd.read_table(path +'\\'  + load + 'A\\H1500' + load +'AU.txt', sep=",")
        data_v = pd.read_table(path +'\\'  + load + 'A\\H1500' + load +'AV.txt', sep=",")
        data_w = pd.read_table(path +'\\'  + load + 'A\\H1500' + load +'AW.txt', sep=",")

        x=1280000 #so that the split below works
        data = pd.concat([data_u[0:x], data_v[0:x], data_w[0:x]], axis=1)
       
        print("\tSplitting")
        data_split = np.array_split(data, len(data)/k)
        
        print("\tWriting to csv")
        to_csv(path, data_split, load, k)
    
'''
Writes already split data into CSV
'''
def to_csv(path, data_split, load, k):
    file = open(path + "\\data" + load +"."  + str(k) + ".txt", "w+")

    #CSV header
    for column in data_split[0]:
        for i in range(0,len(data_split[0])):
            file.write(str(column) + ", ")
    file.write("load"+  "\n")
        
    for split in data_split:
        for column in split:
            for elem in split[column]:
                file.write(str(elem) + ", ")
        file.write(load +  "\n")

'''
Reads already split data and build a file with p samples in total with a uniform dist.

p (int): number of samples in final file
path (String): path to a,ready slpit data
k (int): number of samples of each type in the split data
'''
def sample(path, p, k):
    loads = ['1.5','2.5','3.5','4.5','5.5','6.5','7.5','8.5','9.5','10.5']

    data_sample = pd.DataFrame()
    for load in loads:
        data_case = pd.read_csv(path + '\\data' + load + "." +str(k) + '.txt')
        if (load == '1.5' or load == '5.5' or load =='10.5'):
            data_case_sample = data_case.sample(1334)
        else:
            data_case_sample = data_case.sample(429)
        data_sample = data_sample.append(data_case_sample)
        print('done')
    print(data_sample.shape)
    data_sample.to_csv(path + "\\sample" + str(p) + 'x' + str(k) + '.csv', index=False, header=False)
    

dir_path= "C:\\Users\\Adrian Stetco\Desktop\\1500Load"


for size in [6,8,16,32,64]:
    a= split_data(size, dir_path)
    sample(dir_path, 7000, size)