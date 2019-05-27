#coding=utf-8
import pandas as pd
import numpy as np
import os
from datetime import  datetime
from datetime import  timedelta


action_path = '../jdata/jdata_action.csv'
comment_path = '../jdata/jdata_comment.csv'
product_path = '../jdata/jdata_product.csv'
shop_path = '../jdata/jdata_shop.csv'
user_path = '../jdata/jdata_user.csv'

comment_date = ['2018-03-01', '2018-3-15', '2018-03-30', '2018-04-14', '2018-04-29', '2018-05-15']

def lower_sample_data(df, percent=1):
    '''
    percent:多数类别下采样的数量相对于少数类别样本数量的比例
    '''
    data1 = df[df['label'] == 0]  # 将多数类别的样本放在data1
    data0 = df[df['label'] == 1]  # 将少数类别的样本放在data0
    index = np.random.choice(
        len(data1), size=int(percent*(df.shape[0]-len(data1))),replace=False)  # 随机给定下采样取出样本的序号
    lower_data1 = data1.iloc[list(index)]  # 下采样
    return pd.concat([lower_data1, data0],axis=0)

def date_reduce(date,k):
    date = datetime.strptime(date,'%Y-%m-%d') -timedelta(days=k)
    return datetime.strftime(date,'%Y-%m-%d')

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# 基本用户信息
def get_basic_user_feat(save=True):
    '''
    feature: user_id,age,sex,,user_lv_cd,city_level,province,city,county
    drop: user_reg_tm
    :param save:
    :return:
    '''
    dump_path = '../cache/basic_user.h5'
    if os.path.exists(dump_path):
        user = reduce_mem_usage(pd.read_hdf(dump_path,mode='r',key='user'))
    else:
        user = reduce_mem_usage(pd.read_csv(user_path,sep=','))
        user = user.fillna(0)
        #########  ONE HOT   #####
        # age_df = pd.get_dummies(user['age'],prefix='age')
        # sex_df = pd.get_dummies(user["sex"], prefix="sex")
        # user_lv_df = pd.get_dummies(user["user_lv_cd"], prefix="user_lv_cd")
        # user = pd.concat([user['user_id'], age_df, sex_df, user_lv_df], axis=1)
        ########
        user = user.drop(['user_reg_tm'], axis=1)
        if save:
            user.to_hdf(dump_path,mode='w',key='user')
    return user


# 基本产品信息
def get_basic_product_feat(save=True):
    '''
    feature:  sku_id,brand,shop_id,cate
    :param save:
    :return:
    '''
    dump_path = '../cache/basic_product.h5'
    if os.path.exists(dump_path):
        product = reduce_mem_usage(pd.read_hdf(dump_path,mode='r',key='product'))
    else:
        product = reduce_mem_usage(pd.read_csv(product_path,sep=','))
        ############
        product = product[['sku_id', 'brand', 'shop_id', 'cate']]

        if save:
            product.to_hdf(dump_path,mode='w',key='product')
    return product


# 时间窗口内的活动信息
def get_action(start_date, end_date, save=True):
    '''
    feature: user_id,sku_id,action_time,module_id,type
    :param start_date:
    :param end_date:
    :param save:
    :return:
    '''
    dump_path = '../cache/all_action_%s_%s.h2' % (start_date, end_date)
    if os.path.exists(dump_path):
        action = reduce_mem_usage(pd.read_hdf(dump_path,mode='r',key='action'))
    else:
        action = reduce_mem_usage(pd.read_csv(action_path,sep=','))
        ########
        action = action[(action['action_time'] >= start_date) & (action['action_time'] <= end_date)]
        ##########
        #action.drop(['module_id'], axis=1)
        if save:
            action.to_hdf(dump_path, mode='w',key='action')
    return action




# 时间窗口内不同活动的次数
def get_action_feat(start_date,end_date,save=True):
    dump_path = '../cache/action_feat_%s_%s.h5' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = reduce_mem_usage(pd.read_hdf(dump_path,mode='r',key='af'))
    else:
        actions = get_action(start_date, end_date, save=True)
        actions = actions[['user_id', 'sku_id', 'type']]
        df = pd.get_dummies(actions['type'], prefix='%s-%s-action' % (start_date, end_date))
        actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
        actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
        del actions['type']
        if save:
            actions.to_hdf(dump_path, mode='w',key='af')
    return actions


# 时间窗口内活动的统计信息
def get_accumulate_action_feat(start_date, end_date,save=True):
    dump_path = '../cache/action_accumulate_%s_%s.h5' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = reduce_mem_usage(pd.read_hdf(dump_path,key='aaf',mode='r'))
    else:
        actions = get_action(start_date, end_date,save=True)
        actions = actions.merge(get_basic_product_feat(),on='sku_id',how='left')
        df = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions, df], axis=1) # type: pd.DataFrame
        # 近期行为按时间衰减
        date=datetime.strptime(end_date, '%Y-%m-%d')
        actions['action_time'] = pd.to_datetime(actions['action_time'])
        actions['weights'] = actions['action_time'].apply(lambda x:np.exp(-(date-x).days))
        # actions['weights'] = time.strptime(end_date, '%Y-%m-%d') - actions['datetime']
        # actions['weights'] = actions['weights'].map(lambda x: np.exp(-x.days))
        # print (actions.head(10))
        actions['weights_action_1'] = actions['action_1'] * actions['weights']
        actions['weights_action_2'] = actions['action_2'] * actions['weights']
        actions['weights_action_3'] = actions['action_3'] * actions['weights']
        actions['weights_action_4'] = actions['action_4'] * actions['weights']
        actions['weights_action_5'] = actions['action_5'] * actions['weights']

        del actions['module_id']
        del actions['type']
        del actions['action_time']
        del actions['weights']
        del actions['sku_id']
        actions = actions.groupby(['user_id', 'cate', 'shop_id'], as_index=False).sum()
        actions = actions[['user_id', 'cate', 'shop_id', 'weights_action_1','weights_action_2','weights_action_3','weights_action_4','weights_action_5']]
        if save:
            actions.to_hdf(dump_path, mode='w',key='aaf')
        print('action_feat finish')
    return actions


#
def get_comments_product_feat(start_date, end_date,save=True):
    dump_path = '../cache/comments_accumulate_%s_%s.h5' % (start_date, end_date)
    if os.path.exists(dump_path):
        comments = reduce_mem_usage(pd.read_hdf(dump_path,mode='r', key='cpf'))
        return comments
    else:
        comments = reduce_mem_usage(pd.read_csv(comment_path, sep=','))
        # comment_date_end = end_date
        # comment_date_begin = comment_date[0]
        # for date in reversed(comment_date):
        #     if date < comment_date_end:
        #         comment_date_begin = date
        #         break
        comments = comments[(comments.dt >= start_date) & (comments.dt < end_date)]
        ##
        comments_ac = comments.groupby(['sku_id']).sum().reset_index()[['sku_id',
                                                                         'comments', 'good_comments', 'bad_comments']]

        # comments = comments[(comments.dt >= comment_date_begin) & (comments.dt < comment_date_end)]
        # df = pd.get_dummies(comments['comment_num'], prefix='comment_num')
        # comments = pd.concat([comments, df], axis=1) # type: pd.DataFrame
        # del comments['dt']
        # del comments['comment_num']
        # comments = comments[['sku_id', 'has_bad_comment', 'bad_comment_rate', 'comment_num_1', 'comment_num_2', 'comment_num_3', 'comment_num_4']]
        comments_ac['good_comments_rate'] = comments_ac['good_comments'] / comments_ac['comments']
        comments_ac['bad_comments_rate'] = comments_ac['bad_comments'] / comments_ac['comments']
        comments_ac = comments_ac[['sku_id', 'comments', 'good_comments', 'bad_comments',
                                   'good_comments_rate', 'bad_comments_rate']]
        if save:
            comments_ac.to_hdf(dump_path, mode='w', key='cpf')
    return comments_ac


#################
def get_accumulate_user_feat(start_date, end_date, save=True):
    feature = ['user_id', 'user_action_1_ratio', 'user_action_3_ratio', 'user_action_4_ratio',
               'user_action_5_ratio']
    dump_path = '../cache/user_feat_accumulate_%s_%s.h5' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = reduce_mem_usage(pd.read_hdf(dump_path, mode='r',key='auf'))
    else:
        actions = get_action(start_date, end_date)
        df = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions['user_id'], df], axis=1)
        actions = actions.groupby(['user_id'], as_index=False).sum()
        actions['user_action_1_ratio'] = actions['action_2'] / actions['action_1']
        actions['user_action_3_ratio'] = actions['action_2'] / actions['action_3']
        actions['user_action_4_ratio'] = actions['action_2'] / actions['action_4']
        actions['user_action_5_ratio'] = actions['action_2'] / actions['action_5']

        actions = actions[feature]
        if save:
            actions.to_hdf(dump_path,mode='w',key='auf')
    return actions


def get_accumulate_product_feat(start_date, end_date, save=True):
    feature = ['sku_id', 'product_action_1_ratio', 'product_action_3_ratio', 'product_action_4_ratio',
               'product_action_5_ratio']
    dump_path = '../cache/product_feat_accumulate_%s_%s.h5' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = reduce_mem_usage(pd.read_hdf(dump_path,mode='r',key='apf'))
    else:
        actions = get_action(start_date, end_date)
        df = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions['sku_id'], df], axis=1)
        actions = actions.groupby(['sku_id'], as_index=False).sum()
        actions['product_action_1_ratio'] = actions['action_2'] / actions['action_1']
        actions['product_action_3_ratio'] = actions['action_2'] / actions['action_3']
        actions['product_action_4_ratio'] = actions['action_2'] / actions['action_4']
        actions['product_action_5_ratio'] = actions['action_2'] / actions['action_5']

        actions = actions[feature]
        if save:
            actions.to_hdf(dump_path, mode='w',key='apf')
    return actions

#################
# 品类特征, 调用时候只需要调用get_cate_feat()就好
# [ 'cate', '总浏览数', '总购买数’， '总加购车数', '总关注数', ’总评论数', '1天之内的。。。','3天之内的。。。‘]
def get_cate_feat(start_date, end_date, save=True):
    dump_path = '../cache/cate_feat_accumulate_%s_%s.h5' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = reduce_mem_usage(pd.read_hdf(dump_path, mode='r', key='auf'))
    else:
        product = get_basic_product_feat()[['sku_id', 'cate']]
        actions = get_action(start_date, end_date, save=True)[['sku_id', 'type']]
        actions = pd.merge(actions, product, how='left', on='sku_id')
        df = pd.get_dummies(actions['type'], prefix='cate_%s_%s' % (start_date, end_date))
        actions = pd.concat([actions, df], axis=1).drop(['sku_id', 'type'], axis=1)
        actions = actions.groupby(['cate'], as_index=False).sum()
        for i in (1, 2, 3, 7, 10, 20):
            start_date_tmp = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=i)
            start_date_tmp = start_date_tmp.strftime('%Y-%m-%d')
            tmp_actions = get_cate_time_feat(start_date_tmp, end_date, False)
            # print(tmp_actions.head())
            actions = pd.merge(actions, tmp_actions, how='left', on=['cate'])
        # print(actions.head())
        if save:
            actions.to_hdf(dump_path, mode='w', key='auf')
    return actions

# 按指定时间得到品类特征
def get_cate_time_feat(start_date, end_date, save=True):
    dump_path = '../cache/cate_feat_accumulate_%s_%s.h5' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = reduce_mem_usage(pd.read_hdf(dump_path, mode='r', key='auf'))
    else:
        product = get_basic_product_feat()[['sku_id', 'cate']]
        actions = get_action(start_date, end_date, save=True)[['sku_id', 'type']]
        actions = pd.merge(actions, product, how='left', on='sku_id')
        df = pd.get_dummies(actions['type'], prefix='cate_%s_%s' % (start_date, end_date))
        actions = pd.concat([actions, df], axis=1).drop(['sku_id', 'type'], axis=1)
        actions = actions.groupby(['cate'], as_index=False).sum()
        if save:
            actions.to_hdf(dump_path, mode='w', key='auf')
    return actions


# 用户和品类的交互特征
# [user_id,cate,sku_id,user_cate_1(用户对对品类B的浏览数），user_cate_2(下单数),user_cate_3(关注数),评论数，加购物车数]
def get_accumulate_user_cate_feat(start_date, end_date, save=True):
    dump_path = '../cache/user_cate_feat_accumulate_%s_%s.h5' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = reduce_mem_usage(pd.read_hdf(dump_path, mode='r', key='auf'))
    else:
        # product['sku_id', 'brand', 'shop_id', 'cate']
        product = get_basic_product_feat()[['sku_id','cate']]
        actions = get_action(start_date, end_date, save=True)
        actions = actions[['user_id', 'sku_id', 'type']]
        actions = pd.merge(actions, product, how='left', on='sku_id')
        df = pd.get_dummies(actions['type'], prefix='user_cate_%s_%s' % (start_date, end_date))
        actions = pd.concat([actions, df], axis=1).drop('sku_id', axis=1)
        actions = actions.groupby(['user_id', 'cate'], as_index=False).sum()
        for i in (1, 2, 3, 7, 10, 20):
            start_date_tmp = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=i)
            start_date_tmp = start_date_tmp.strftime('%Y-%m-%d')
            tmp_actions = get_user_cate_feat(start_date_tmp, end_date,False)
            actions = pd.merge(actions, tmp_actions, how='left', on=['user_id', 'cate'])
        del actions['type']

        # print(actions.head())
        print(actions.columns)
        if save:
            actions.to_hdf(dump_path, mode='w', key='auf')
    return actions


#按指定时间得到用户和品类的特征表
def get_user_cate_feat(start_date, end_date, save=True):
    dump_path = '../cache/user_cate_feat_%s_%s.h5' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = reduce_mem_usage(pd.read_hdf(dump_path, mode='r', key='auf'))
    else:
        # product['sku_id', 'brand', 'shop_id', 'cate']
        product = get_basic_product_feat()[['sku_id', 'cate']]
        actions = get_action(start_date, end_date, save=True)
        actions = actions[['user_id', 'sku_id', 'type']]
        actions = pd.merge(actions, product, how='left', on='sku_id')
        df = pd.get_dummies(actions['type'], prefix='user_cate_%s_%s' % (start_date, end_date))
        actions = pd.concat([actions, df], axis=1).drop('sku_id', axis=1)
        actions = actions.groupby(['user_id', 'cate'], as_index=False).sum()
        del actions['type']
        if save:
            actions.to_hdf(dump_path, mode='w', key='auf')
    return actions

#####################
# 用户对商品点击、收藏、加购物车、购买量除以用户点击、收藏、加购物车、购买量
def get_cross_feat_5(start_date, end_date, save=True):
    dump_path = '../cache/cross_feat_5_%s_%s.h5' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = reduce_mem_usage(pd.read_hdf(dump_path, mode='r', key='auf'))
    else:
        user_acc = get_user_acc_feat(start_date, end_date, save=True)
        actions = get_action(start_date, end_date, save=True)[['user_id', 'type']]
        df = pd.get_dummies(actions['type'], prefix='user_zong_type')
        actions = pd.concat([actions, df], axis=1).drop(['type'], axis=1)
        actions = actions.groupby(['user_id'], as_index=False).sum()
        actions = pd.merge(user_acc, actions, how='left', on='user_id')
        actions['u1_z1_ratio'] = actions['user_type_1'] / actions['user_zong_type_1']
        actions['u2_z2_ratio'] = actions['user_type_2'] / actions['user_zong_type_2']
        actions['u3_z3_ratio'] = actions['user_type_3'] / actions['user_zong_type_3']
        actions['u4_z4_ratio'] = actions['user_type_4'] / actions['user_zong_type_4']
        actions['u5_z5_ratio'] = actions['user_type_5'] / actions['user_zong_type_5']
        actions = actions.drop(
            ['user_zong_type_1', 'user_zong_type_2', 'user_zong_type_3', 'user_zong_type_4', 'user_zong_type_5', 'user_type_1', 'user_type_2',
             'user_type_3', 'user_type_4', 'user_type_5'], axis=1)
        if save:
            actions.to_hdf(dump_path, mode='w', key='auf')
    return actions

# 用户对商品点击、收藏、加购物车、购买量除以用户对该类所有商品点击、收藏、加购物车、购买量
def get_cross_feat_6(start_date, end_date, save=True):
    dump_path = '../cache/cross_feat_6_%s_%s.h5' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = reduce_mem_usage(pd.read_hdf(dump_path, mode='r', key='auf'))
    else:
        user_acc = get_user_acc_feat(start_date, end_date, save=True)
        product = get_basic_product_feat()[['sku_id', 'cate']]
        actions = get_action(start_date, end_date, save=True)[['user_id', 'sku_id', 'type']]
        actions = pd.merge(actions, product, how='left', on='sku_id')
        df = pd.get_dummies(actions['type'], prefix='user_cate')
        actions = pd.concat([actions, df], axis=1).drop('sku_id', axis=1)
        user_cate_acc = actions.groupby(['user_id', 'cate'], as_index=False).sum()
        user_cate_acc = user_cate_acc.drop(['type', 'cate'], axis=1)
        actions = pd.merge(user_acc, user_cate_acc, how='left', on='user_id')
        actions['u1_c1_ratio'] = actions['user_type_1'] / actions['user_cate_1']
        actions['u2_c2_ratio'] = actions['user_type_2'] / actions['user_cate_2']
        actions['u3_c3_ratio'] = actions['user_type_3'] / actions['user_cate_3']
        actions['u4_c4_ratio'] = actions['user_type_4'] / actions['user_cate_4']
        actions['u5_c5_ratio'] = actions['user_type_5'] / actions['user_cate_5']
        actions = actions.drop(
            ['user_cate_1', 'user_cate_2', 'user_cate_3', 'user_cate_4', 'user_cate_5',
             'user_type_1', 'user_type_2',
             'user_type_3', 'user_type_4', 'user_type_5'], axis=1)
        print(actions.head())
    #     if save:
    #         actions.to_hdf(dump_path, mode='w', key='auf')
    # return actions
# 用户对商品点击、收藏、加购物车、购买量除以该商品被点击、收藏、加购物车、购买量
def get_cross_feat_7(start_date, end_date, save=True):
    dump_path = '../cache/cross_feat_7_%s_%s.h5' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = reduce_mem_usage(pd.read_hdf(dump_path, mode='r', key='auf'))
    else:
        user_acc = get_user_acc_feat(start_date, end_date, save=True)
        actions = get_action(start_date, end_date, save=True)[['sku_id', 'type']]
        df = pd.get_dummies(actions['type'], prefix='sku_type')
        actions = pd.concat([actions, df], axis=1).drop(['type'], axis=1)
        product_acc = actions.groupby(['sku_id'], as_index=False).sum()
        actions = pd.merge(user_acc, product_acc, how='left', on='sku_id')
        actions['u1_p1_ratio'] = actions['user_type_1']/actions['sku_type_1']
        actions['u2_p2_ratio'] = actions['user_type_2']/actions['sku_type_2']
        actions['u3_p3_ratio'] = actions['user_type_3']/actions['sku_type_3']
        actions['u4_p4_ratio'] = actions['user_type_4']/actions['sku_type_4']
        actions['u5_p5_ratio'] = actions['user_type_5']/actions['sku_type_5']
        actions = actions.drop(
            ['sku_type_1', 'sku_type_2', 'sku_type_3', 'sku_type_4', 'sku_type_5', 'user_type_1', 'user_type_2',
             'user_type_3', 'user_type_4', 'user_type_5'], axis=1)
        # print(actions.head())
        if save:
            actions.to_hdf(dump_path, mode='w', key='auf')
    return actions

# 商品被点击、收藏、加购物车、购买量除以该类商品被点击、收藏、加购物车、购买量
def get_cross_feat_8(start_date, end_date, save=True):
    dump_path = '../cache/cross_feat_8_%s_%s.h5' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = reduce_mem_usage(pd.read_hdf(dump_path, mode='r', key='auf'))
    else:
        product = get_basic_product_feat()[['sku_id', 'cate']]
        actions = get_action(start_date, end_date, save=True)[['sku_id', 'type']]
        actions = pd.merge(actions, product, how='left', on='sku_id')
        df1 = pd.get_dummies(actions['type'], prefix='sku_type')
        cate_acc = get_cate_acc_feat(start_date, end_date, save=True)
        actions = pd.concat([actions, df1], axis=1).drop(['type', 'cate'], axis=1)
        product_acc = actions.groupby(['sku_id'], as_index=False).sum()
        merge_1 = pd.merge(product, product_acc, how='left', on='sku_id')
        merge_all = pd.merge(merge_1, cate_acc, how='left', on='cate')
        merge_all['p1_c1_ratio'] = merge_all['sku_type_1'] / merge_all['cate_type_1']
        merge_all['p2_c2_ratio'] = merge_all['sku_type_2'] / merge_all['cate_type_2']
        merge_all['p3_c3_ratio'] = merge_all['sku_type_3'] / merge_all['cate_type_3']
        merge_all['p4_c4_ratio'] = merge_all['sku_type_4'] / merge_all['cate_type_4']
        merge_all['p5_c5_ratio'] = merge_all['sku_type_5'] / merge_all['cate_type_5']
        actions = merge_all.drop(['sku_type_1', 'sku_type_2', 'sku_type_3', 'sku_type_4','sku_type_5', 'cate_type_1', 'cate_type_2',
                                  'cate_type_3', 'cate_type_4', 'cate_type_5'], axis=1)
        print(actions.head())
    #     if save:
    #         actions.to_hdf(dump_path, mode='w', key='auf')
    # return actions


def get_cate_acc_feat(start_date, end_date, save=True):
    dump_path = '../cache/cate_acc_accumulate_%s_%s.h5' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = reduce_mem_usage(pd.read_hdf(dump_path, mode='r', key='auf'))
    else:
        product = get_basic_product_feat()[['sku_id', 'cate']]
        actions = get_action(start_date, end_date, save=True)[['sku_id', 'type']]
        actions = pd.merge(actions, product, how='left', on='sku_id')
        df = pd.get_dummies(actions['type'], prefix='cate_type')
        actions = pd.concat([actions, df], axis=1).drop(['sku_id', 'type'], axis=1)
        actions = actions.groupby(['cate'], as_index=False).sum()
        if save:
            actions.to_hdf(dump_path, mode='w', key='auf')
    return actions

def get_user_acc_feat(start_date, end_date, save=True):
    dump_path = '../cache/user_acc_accumulate_%s_%s.h5' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = reduce_mem_usage(pd.read_hdf(dump_path, mode='r', key='auf'))
    else:
        actions = get_action(start_date, end_date, save=True)[['user_id', 'sku_id', 'type']]
        df = pd.get_dummies(actions['type'], prefix='user_type')
        actions = pd.concat([actions, df], axis=1).drop(['type'], axis=1)
        actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
        # actions = actions.drop('sku_id', axis=1)
        if save:
            actions.to_hdf(dump_path, mode='w', key='auf')
        return actions





#################
def get_labels(start_date, end_date, days=20, save=True):
    dump_path = '../cache/labels_%s_%s.h5' % (start_date, end_date)
    if os.path.exists(dump_path):
        sub = reduce_mem_usage(pd.read_hdf(dump_path,key='labels'))
    else:
        product = get_basic_product_feat()
        start_days = datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=days)
        start_days = start_days.strftime('%Y-%m-%d')
        actions = get_action(start_days, end_date)
        pre = pd.merge(actions, product, on='sku_id', how='inner')[['user_id', 'sku_id', 'action_time', 'type']]
        # 期间购买用户
        sub = pre[(pre['type'] == 2) & (pre['action_time'] >= start_date) & (pre['action_time'] <= end_date)][['user_id','sku_id']].drop_duplicates()
        sub['label'] = 1
        # 候选用户
        actions_pre = actions[(actions['action_time'] >= start_days) & (actions['action_time'] <= start_date)][['user_id','sku_id']].drop_duplicates()
        sub = sub.merge(actions_pre,how='right',on=['user_id','sku_id'])
        sub=sub.fillna(0)
        sub = sub[['user_id', 'sku_id', 'label']]
        # sub = lower_sample_data(sub,1)
        if save:
            sub.to_hdf(dump_path,mode='w',key='labels')
    return sub

###########

def make_train_set(train_start_date, train_end_date, test_start_date, test_end_date, days=30):
    dump_path = '../cache/train_set_%s_%s_%s_%s.h5' % (train_start_date, train_end_date, test_start_date, test_end_date)
    if os.path.exists(dump_path):
        actions = reduce_mem_usage(pd.read_hdf(dump_path,key='train'))
    else:
        start_days = "2018-02-01"
        user = get_basic_user_feat()
        product = get_basic_product_feat()
        # 用户的行为转化率
        user_acc = get_accumulate_user_feat(start_days, train_end_date)
        # 基于sku_id的购买转化率
        product_acc = get_accumulate_product_feat(start_days, train_end_date)
        # 基于sku_id的评论数据
        comment_acc = get_comments_product_feat(train_start_date, train_end_date)
        # 基于user,cate,shop的权重行为
        action_feat = get_accumulate_action_feat(train_start_date,train_end_date)
        user_cate_feat = get_accumulate_user_cate_feat(train_start_date,train_end_date)
        labels = get_labels(test_start_date, test_end_date)

        # generate 时间窗口
        # actions = get_accumulate_action_feat(train_start_date, train_end_date)
        actions = labels
        for i in (1, 2, 3, 5, 7, 10, 15, 21, 30):
            start_days = date_reduce(train_end_date,i)
            # 用户对商品的操作
            actions = pd.merge(actions, get_action_feat(start_days, train_end_date), how='left',on=['user_id', 'sku_id'])
            # actions = pd.merge(actions,)
        actions = pd.merge(actions, product, how='left', on='sku_id')
        # actions = actions.drop_duplicates(['user_id', 'cate', 'shop_id'])
        actions = pd.merge(actions, user, how='left', on='user_id')
        actions = pd.merge(actions, user_acc, how='left', on='user_id')
        actions = pd.merge(actions, product_acc, how='left', on='sku_id')
        actions = pd.merge(actions, comment_acc, how='left', on='sku_id')
        actions = pd.merge(actions, user_cate_feat, how='left', on=['user_id', 'cate'])
        actions = pd.merge(actions, action_feat, how='left', on=['user_id', 'cate', 'shop_id'])
        actions = actions.fillna(0)

        actions[['user_id', 'cate', 'shop_id', 'brand']] = actions[['user_id', 'cate', 'shop_id', 'brand']].astype(
            'int')
        actions.to_hdf(dump_path,key='train')

    users = actions[['user_id', 'cate', 'shop_id']].astype('int')
    labels = actions['label'].astype('int')
    # del actions['user_id']
    del actions['sku_id']
    # del actions['cate']
    # del actions['shop_id']
    del actions['label']

    return users, actions, labels


def report(pred, label):

    actions = label
    result = pred

    # 所有用户商品对
    all_user_item_pair = actions['user_id'].map(str) + '-' + actions['sku_id'].map(str)
    all_user_item_pair = np.array(all_user_item_pair)
    # 所有购买用户
    all_user_set = actions['user_id'].unique()

    # 所有品类中预测购买的用户
    all_user_test_set = result['user_id'].unique()
    all_user_test_item_pair = result['user_id'].map(str) + '-' + result['sku_id'].map(str)
    all_user_test_item_pair = np.array(all_user_test_item_pair)

    # 计算所有用户购买评价指标
    pos, neg = 0,0
    for user_id in all_user_test_set:
        if user_id in all_user_set:
            pos += 1
        else:
            neg += 1
    all_user_acc = 1.0 * pos / ( pos + neg)
    all_user_recall = 1.0 * pos / len(all_user_set)
    print('所有用户中预测购买用户的准确率为 ' + str(all_user_acc))
    print('所有用户中预测购买用户的召回率' + str(all_user_recall))

    pos, neg = 0, 0
    for user_item_pair in all_user_test_item_pair:
        if user_item_pair in all_user_item_pair:
            pos += 1
        else:
            neg += 1
    all_item_acc = 1.0 * pos / ( pos + neg)
    all_item_recall = 1.0 * pos / len(all_user_item_pair)
    print( '所有用户中预测购买商品的准确率为 ' + str(all_item_acc))
    print('所有用户中预测购买商品的召回率' + str(all_item_recall))
    F11 = 6.0 * all_user_recall * all_user_acc / (5.0 * all_user_recall + all_user_acc)
    F12 = 5.0 * all_item_acc * all_item_recall / (2.0 * all_item_recall + 3 * all_item_acc)
    score = 0.4 * F11 + 0.6 * F12
    print('F11=' + str(F11))
    print('F12=' + str(F12))
    print('score=' + str(score))



def Recall(df_true_pred,real_buy, threshold, flag):
    """
        df_true_pred : 'user_id', 'cate', 'shop_id', 'label', 'pred_prob'
            flag : 'user_cate' or 'user_cate_shop'
            Threshold = 0.5
            Recall = TP / (TP + FN)
    """
    temp_ = df_true_pred[df_true_pred['pred_prob']>=threshold][['user_id','cate','shop_id']]
    if flag == 'user_cate':
        temp_ = temp_[['user_id','cate']].drop_duplicates(['user_id', 'cate'])
        temp_tp = temp_.merge(real_buy[['user_id','cate']],how='inner',on=['user_id','cate']).drop_duplicates(['user_id','cate'])
        recall = temp_tp.shape[0] * 1.0 / real_buy.shape[0]
    elif flag == 'user_cate_shop':
        temp_ = temp_.drop_duplicates(['user_id','cate','shop_id'])
        temp_tp = temp_.merge(real_buy,how='inner',on=['user_id','cate','shop_id']).drop_duplicates(['user_id','cate','shop_id'])
        recall = temp_tp.shape[0] * 1.0 / real_buy.shape[0]
    else:
        recall = -1
    return recall

def Precision(df_true_pred,real_buy, threshold, flag):
    """
        df_true_pred : 'user_id', 'cate', 'shop_id', 'label', 'pred_prob'
            flag : 'user_cate' or 'user_cate_shop'
            Threshold
            Precision = TP / (TP + FP)
    """
    temp_ = df_true_pred[df_true_pred['pred_prob'] >= threshold][['user_id', 'cate', 'shop_id']]
    if flag == 'user_cate':
        temp_ = temp_[['user_id', 'cate']].drop_duplicates(['user_id', 'cate'])
        temp_tp = temp_.merge(real_buy[['user_id', 'cate']], how='inner', on=['user_id', 'cate']).drop_duplicates(
            ['user_id', 'cate'])
        precision = temp_tp.shape[0] * 1.0 / temp_.shape[0]
    elif flag == 'user_cate_shop':
        temp_ = temp_.drop_duplicates(['user_id', 'cate', 'shop_id'])
        temp_tp = temp_.merge(real_buy, how='inner', on=['user_id', 'cate', 'shop_id']).drop_duplicates(
            ['user_id', 'cate', 'shop_id'])
        precision = temp_tp.shape[0] * 1.0 / temp_.shape[0]
    else:
        precision = -1
    return precision

def get_metrice(df_true_pred, threshold,real_buy):
    """
        df_true_pred : 'user_id', 'cate', 'shop_id', 'label', 'pred_prob'
        Threshold = 0.5
    """
    # real_buy = reduce_mem_usage(get_action('2018-04-09','2018-04-16'))
    # real_buy = real_buy.merge(get_basic_product_feat()[['sku_id','cate','shop_id']],on='sku_id',how='inner')
    # # print(real_buy.columns)
    # real_buy = real_buy[real_buy['type']==2][['user_id','cate','shop_id']]
    # 用户-品类
    R1_1 = Recall(df_true_pred,real_buy, threshold, flag='user_cate')
    P1_1 = Precision(df_true_pred,real_buy, threshold, flag='user_cate')
    F1_1 = 3 * R1_1 * P1_1 / (2 * R1_1 + P1_1)

    # 用户-品类-店铺
    R1_2 = Recall(df_true_pred,real_buy, threshold, flag='user_cate_shop')
    P1_2 = Precision(df_true_pred,real_buy, threshold, flag='user_cate_shop')
    F1_2 = 5 * R1_2 * P1_2 / (2 * R1_2 + 3 * P1_2)

    # 总分

    score = 0.4 * F1_1 + 0.6 * F1_2
    print('F11 %s   F12 %s   score %s' % (F1_1, F1_2,score))
    return score

def count_submission(start_date, end_date):
    actions = get_action(start_date, end_date, save=True)[['user_id', 'sku_id', 'type']]
    product = get_basic_product_feat(save=True)[['sku_id', 'shop_id', 'cate']]
    actions = pd.merge(actions, product, how='left', on='sku_id')
    # print(product.shape[0])
    # print(actions.shape[0])
    buy1 = actions[actions.type == 4]
    #买过的记录
    print('最后7天的购买记录一共有', buy1.shape[0])
    #买过的不重复的人
    print('其中在最后7天不重复的购买人数', buy1.user_id.nunique())
    buy = buy1.groupby(['user_id', 'cate'], as_index=False).sum()
    print('在最后7天user-cate-shop的记录一共有',np.shape(buy)[0])
    print( buy.head())




if __name__ == '__main__':
    train_start_date = '2018-03-02'
    train_end_date = '2018-04-09'
    test_start_date = '2018-03-02'
    test_end_date = '2018-04-16'
    # user, action, label = make_train_set(train_start_date, train_end_date, test_start_date, test_end_date)
    # print(user.head(10))
    # print(action.head(10))
    # count_submission(train_start_date, train_end_date)
    get_cross_feat_8(train_start_date, train_end_date, save=True)
    # get_user_acc_feat(train_start_date, train_end_date, save=True)