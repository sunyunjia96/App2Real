import torch
import numpy as np
import os

# wipe off the outliers
# type: std - wipe off data points without the mean+_3*std
# TBD: type: IQR - wipe off data points without (Q1 - 1.5 * IQR) or (Q3 + 1.5 * IQR)
def get_clean_avg_one_dim(input_list, type='std'):
    if len(input_list) <= 2:
        return np.mean(input_list)
    if type=='std':
        std = np.std(input_list, ddof=1)
        mean = np.mean(input_list)
        if std==0:
            return mean
        lower_bound = mean-1.5*std
        upper_bound = mean+1.5*std
    new_list = []
    for item in input_list:
        if lower_bound < item < upper_bound:
            new_list.append(item)
    return np.mean(new_list)

def get_clean_avg(input_list):
    input_list = input_list.tolist()
    clean_avg = []
    for i in range(5):
        clean_avg.append(get_clean_avg_one_dim(np.array(input_list)[:,i]))
    return clean_avg

# iteration algorithm
# to give data points that are far away from the initial average less weight
def get_weighted_avg_one_dim(input_list, start):
    if start=='default':
        w_avg = np.mean(input_list)
    elif start=='clean':
        w_avg = get_clean_avg_one_dim(input_list)
    for i in range(30):
        dist = (input_list - w_avg)**2
        # get un-normalized weights
        w = 1/(dist+1e-6)
        normalized_factor = np.sum(w)
        # normalized weights
        w = w/normalized_factor
        w_avg_new = np.sum(input_list*w)
        if np.abs(w_avg_new-w_avg) < 0.001:
            return w_avg_new
        w_avg = w_avg_new
    return w_avg_new

def get_weighted_avg(input_list, start='default'):
    input_list = input_list.tolist()
    traits_w_avg = []
    for i in range(5):
        traits_w_avg.append(get_weighted_avg_one_dim(np.array(input_list)[:,i],start=start))
    return traits_w_avg

def get_all_checkpoints(root_dir):
    path_list = []
    dir_list = []
    json_list = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # 检查文件后缀
            if filename.endswith('.pth.tar'):
                # 获取文件绝对路径并添加到列表
                file_path = os.path.abspath(os.path.join(dirpath, filename))
                relative_path = os.path.relpath(file_path, root_dir)
                path_list.append(relative_path)
                # 获得建json所需路径
                dir_path = '/'.join(relative_path.split('/')[:-2])
                dir_list.append(dir_path)
                json_path = '/'.join(relative_path.split('/')[:-1])+'.json'
                json_list.append(json_path)
    return path_list, dir_list, json_list

if __name__ == '__main__':
    path_list, dir_list, json_list = get_all_checkpoints('checkpoint/repeat_exp/')
    print(path_list)
    print(dir_list)
    print(json_list)