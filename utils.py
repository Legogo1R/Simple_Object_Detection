import os
import pandas as pd



def get_dir_of_obj(object, filter='__', include=False):
    """
    Returns methods and atributes of object with filtering
    """
    method_list = [method for method in dir(object) if method.startswith(filter) is include]
    return method_list


def dataframe_from_file(file_path):
    """
    Creates pandas dataframe from YOLOv8 dataset format
    """
    path2file = os.path.join(file_path, 'labels')
    df = pd.DataFrame()
    file_list = []
    col_names=['class', 'x1', 'x2', 'y1', 'y2','img_name']

    for file in os.listdir(path2file):
        if file[-4:] == '.txt':
            file_list.append(file)

    for file_name in file_list:
        with open(os.path.join(path2file, file_name), encoding='utf-8') as txt:
            for row in txt:
                row = row[:-1]  # Remove \n
                row = row.split(" ")
                row = [x if x != '' else None for x in row]
                # print(row)
                new_df = pd.concat([pd.DataFrame([row],dtype='float'),pd.DataFrame([file_name])],axis=1)  # Concat labels and file name
                df = pd.concat([df, new_df], axis=0)  # Append new rows

    df.columns = col_names
    return df
