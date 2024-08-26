'''
Author: your name
Date: 2021-03-16 17:59:40
LastEditTime: 2021-04-23 10:08:52
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /NashHE/code/merge_csv.py
'''
import os, csv
import pandas as pd

dir_path = '/Users/zhanghangsheng/my_code/NashHE/data/CIC-DoHBrw-2020'
file_out = "/Users/zhanghangsheng/my_code/NashHE/data/CIC-DoHBrw-2020/ContagioPDFData.csv"

file_count = 0
file_list = []
for root, dirs, files in os.walk(dir_path, topdown=True):
    for file in files:
        if file[-3:]=='csv':
            file_path = os.path.join(root, file)
            file_list.append(file_path)
            print(file_path.split('/')[-2])
            # check they have the same header
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                row1 = next(reader)
                print(row1)
            f.close()
            file_count += 1
print(file_count)


def pd_process(file):
    file_path = os.path.join(root, file)
    label=file_path.split('/')[-2]
    table=pd.read_csv(file)
    table['label']=label
    return table
    
def produceOneCSV(list_of_files, file_out):
    """
     Function:
      Produce a single CSV after combining all files
    """
    # # Consolidate all CSV files into one object
    result_obj = pd.concat([pd_process(file) for file in list_of_files])
    # # Convert the above object into a csv file and export
    result_obj.to_csv(file_out, index=False, encoding="utf-8")

produceOneCSV(file_list, file_out)
