#-*-coding:utf-8-*-
"""
author:zhangdiandian
lfm model train main fuction
"""
import numpy as np
import sys
from util import read

sys.path.append("../util")#引入工具函数

import operator

def lfm_train(train_data,F,alpha,beta,step):
    """

    :param train_data: train_data for lfm
    :param F: user vector len,item vector len
    :param alpha: regularization factor
    :param beta: learning rate
    :param step: iteration num
    :return: dict: key itemid,value :list
            dict :key userid,value：list
    """
    user_vec = {}
    item_vec = {}
    for step_index in range(step):
        print(step_index,'/',step)
        for data_instance in train_data:
            userid,itemid,label = data_instance
            if userid not in user_vec:
                user_vec[userid] = init_model(F) #初始化，F个参数，标准正态分布
            if itemid not in item_vec:
                item_vec[itemid] = init_model(F) #初始化
            #模型迭代部分
            delta = label - model_predict(user_vec[userid],item_vec[itemid]) #预测与实际的差
            for index in range(F):#index就是f
                #出于工程角度的考虑，这里没有照搬公式的2倍，而是1倍，效果是一样的
                user_vec[userid][index] += beta*(delta*item_vec[itemid][index] - alpha*user_vec[userid][index])#这里没有照搬ppt的2倍
                item_vec[itemid][index] += beta*(delta*user_vec[userid][index] - alpha*item_vec[itemid][index])

            beta = beta * 0.9 #参数衰减，目的是：在每一轮迭代的时候，接近收敛时，能慢一点
    return  user_vec,item_vec

def init_model(vector_len):
    """
    :param vector_len: the len of vector
    :return: a ndarray
    """
    return np.random.randn(vector_len)

def model_predict(user_vector,item_vector):
    """
    distance between user_vector and item_vector
    :param user_vector: model produce user vector
    :param item_vector: model produce item vector
    :return: a num
    """
    res = np.dot(user_vector,item_vector)/(np.linalg.norm(user_vector)*np.linalg.norm(item_vector))
    return res

def model_train_process():
    """
    test lfm model train
    :return:
    """
    # train_data = read.get_train_data("../data/ml-1m/ratings.txt")
    train_data = read.get_train_data("../data/ratings.txt")
    user_vec,item_vec = lfm_train(train_data,50,0.01,0.1,50)
    recom_list = give_recom_result(user_vec,item_vec,'4')
    print(recom_list)
    ana_recom_result(train_data,'4',recom_list)

    return user_vec,item_vec

def give_recom_result(user_vec,item_vec,userid):
    """
    use lfm model result give fix userid recom result
    :param user_vec: model result
    :param item_vec: model result
    :param userid:fix userid
    :return:a list:[(itemid,score)(itemid1,score1)]
    """
    fix_num = 5 #排序，推荐前fix_num个结果
    if userid not in user_vec:
        return []
    record = {}
    recom_list = []
    user_vector = user_vec[userid]
    for itemid in item_vec:
        item_vector = item_vec[itemid]
        res = np.dot(user_vector,item_vector)/(np.linalg.norm((user_vector)*np.linalg.norm(item_vector))) #余弦距离
        record[itemid] = res
        record_list = list(record.items())
        #排序
    for zuhe in sorted(record.items(), key=lambda rec: record_list[1], reverse=True)[:fix_num]:
        itemid = zuhe[0]
        score = round(zuhe[1],3)
        recom_list.append((itemid,score))
    return recom_list

def  ana_recom_result(train_data,userid,recom_list):
    """
    debug recom result for userid
    :param train_data: train data for userid
    :param userid: fix userid
    :param recom_list: recom result by lfm
    :return: no return
    """
    # item_info = read.get_item_info("../data/ml-1m/movies.txt")
    item_info = read.get_item_info("../data/movies.txt")
    print("该用户曾给过好评的电影如下：")
    for data_instance in train_data:
        tmp_userid,itemid,label = data_instance
        if tmp_userid == userid and label == 1:
            print(item_info[itemid])
    print("前n个推荐电影为：")
    cnt = 1
    for zuhe in recom_list:
        print(cnt,item_info[zuhe[0]])
        cnt += 1

# def test_


if __name__ == '__main__':
    model_train_process()
