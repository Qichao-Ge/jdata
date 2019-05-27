import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from tract_feat import make_train_set
import xgboost as xgb
from tract_feat import get_metrice,reduce_mem_usage,get_action,get_basic_product_feat
import numpy as np



def lgbm():
    train_start_date = '2018-03-01'
    train_end_date = '2018-04-09'
    test_start_date = '2018-04-09'
    test_end_date = '2018-04-16'

    sub_start_date = '2018-03-08'
    sub_end_date = '2018-04-16'
    sub_test_start_date = '2018-04-16'
    sub_test_end_date = '2018-04-23'

    user_index, training_data, label = make_train_set(train_start_date, train_end_date, test_start_date, test_end_date)
    sub_user_index, sub_trainning_date, sub_label = make_train_set(sub_start_date, sub_end_date,
                                                                   sub_test_start_date, sub_test_end_date)

    X_train_gbm, X_test_gbm, y_train_gbm, y_test_gbm = train_test_split(training_data, label, test_size=0.2, random_state=4,stratify=label,shuffle=True)

    train_metrics = pd.concat([user_index, label], axis=1)

    submit_ID = sub_user_index

    params = {
        'is_unbalance': 'true',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': ['auc',"binary_logloss"],
        'learning_rate': 0.01,
        'num_leaves': 31,
        'min_child_samples': 50,
        'max_bin': 100,
        'subsample': 0.7,
        'subsample_freq': 1,
        'colsample_bytree': 0.7,
        'min_child_weight': 0,
        'seed': 2019,
        'nthread': 4,
        'verbose': 0,

    }


    lgb_train = lgb.Dataset(X_train_gbm, label=y_train_gbm, categorical_feature=['user_id','cate','shop_id','brand']
                            )
    lgb_eval = lgb.Dataset(X_test_gbm,label= y_test_gbm, reference=lgb_train, categorical_feature=['user_id','cate','shop_id','brand']
                           )
    # params={'num_leaves': 31,
    #         'num_trees': 200,
    #         'objective': 'binary',
    #         'metric':'auc'}

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=2000,
                    valid_sets=[lgb_eval, lgb_train],
                    early_stopping_rounds=200)
    y_train_predict = gbm.predict(training_data, num_iteration=gbm.best_iteration)
    y_predict = gbm.predict(sub_trainning_date, num_iteration=gbm.best_iteration)
    # 线下验证公式 ('user_id', 'cate', 'shop_id', 'label', 'pred_prob')
    train_metrics['pred_prob'] = y_train_predict.reshape(-1, 1)
    submit_ID['pred_prob'] = y_predict.reshape(-1, 1)

    scores = 0
    train_metrics = train_metrics.groupby(['user_id', 'cate', 'shop_id'], as_index = False).sum()
    train_metrics = train_metrics.sort_values(by='pred_prob', ascending=False)
    # train_metrics.drop_duplicates(['user_id','cate','shop_id'])
    train_metrics = train_metrics.reset_index()
    size = train_metrics.shape[0]
    real_buy = reduce_mem_usage(get_action('2018-04-09','2018-04-16'))
    real_buy = real_buy.merge(get_basic_product_feat()[['sku_id','cate','shop_id']],on='sku_id',how='inner')
    real_buy = real_buy[real_buy['type'] == 2][['user_id', 'cate', 'shop_id']]
    for i in [int(i*size*0.001-1) for i in range(1, 1000)]:
        threshold = train_metrics['pred_prob'].iloc[i]
        scores_temp = get_metrice(train_metrics, threshold=threshold,real_buy=real_buy)
        print(str(i) + '  ' + str(scores_temp))
        if scores_temp > scores:
            scores = scores_temp
            fin_threshold = threshold
    print('inline score ' + str(scores) + 'threshold' + str(fin_threshold))


    submit_ID['pred_prob'] = y_predict.reshape(-1,1)
    submit_ID = submit_ID.groupby(['user_id', 'cate', 'shop_id'], as_index=False).sum()
    submit_ID = submit_ID.sort_values(by='pred_prob', ascending=False)
    submit_ID = submit_ID.reset_index()
    submit_ID['pred_prob'] = np.log(submit_ID['pred_prob'])
    submit_ID.to_csv('score.csv', header=True, sep=',', index=0)
    submit_ID = submit_ID[submit_ID['pred_prob']>=np.log(fin_threshold)]
    print('final submit number %s'% (submit_ID.shape[0]))
    submit_ID[['user_id', 'cate', 'shop_id']].astype(int).to_csv(
        '2019_5_21_lgbm.csv', header=True, sep=',', encoding='utf-8', index=0)

def lgbm_cate():
    train_start_date = '2018-03-01'
    train_end_date = '2018-04-09'
    test_start_date = '2018-04-09'
    test_end_date = '2018-04-16'

    sub_start_date = '2018-03-08'
    sub_end_date = '2018-04-16'
    sub_test_start_date = '2018-04-16'
    sub_test_end_date = '2018-04-23'

    user_index, training_data, label = make_train_set(train_start_date, train_end_date, test_start_date, test_end_date)

    sub_user_index, sub_trainning_date, sub_label = make_train_set(sub_start_date, sub_end_date,
                                                                   sub_test_start_date, sub_test_end_date)
    train_metrics = pd.concat([user_index, label], axis=1)

    train_metrics['pred_prob'] = 0
    submit_ID = sub_user_index
    submit_ID['pred_prob'] = 0

    lgbm_list={}
    for i in list(set(training_data['cate']).intersection(set(sub_trainning_date['cate']))):
        if label[training_data['cate']==i].nunique() == 1:
            print('cate '+str(i)+' pass')
            continue
        print('training cate '+str(i))
        X_train_gbm, X_test_gbm, y_train_gbm, y_test_gbm = train_test_split(training_data[training_data['cate']==i],
                        label[training_data['cate']==i], test_size=0.2, random_state=4,stratify=label[training_data['cate']==i],shuffle=True)


        params = {
            'is_unbalance': 'true',
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.01,
            'num_leaves': 31,
            'min_child_samples': 50,
            'max_bin': 100,
            'subsample': 0.7,
            'subsample_freq': 1,
            'colsample_bytree': 0.7,
            'min_child_weight': 0,
            'seed': 2019,
            'nthread': 4,
            'verbose': 0,

        }


        lgb_train = lgb.Dataset(X_train_gbm, label=y_train_gbm, categorical_feature=['user_id','cate','shop_id','brand']
                                )
        lgb_eval = lgb.Dataset(X_test_gbm,label= y_test_gbm, reference=lgb_train, categorical_feature=['user_id','cate','shop_id','brand']
                               )
        # params={'num_leaves': 31,
        #         'num_trees': 200,
        #         'objective': 'binary',
        #         'metric':'auc'}

        lgbm_list[i] = lgb.train(params,
                        lgb_train,
                        num_boost_round=2000,
                        valid_sets=[lgb_eval,lgb_train],
                        early_stopping_rounds=200)


    for key,values in lgbm_list.items():
        y_train_predict = values.predict(training_data[training_data['cate']==key], num_iteration=values.best_iteration)
        y_predict = values.predict(sub_trainning_date[sub_trainning_date['cate']==key], num_iteration=values.best_iteration)
        # 线下验证公式 ('user_id', 'cate', 'shop_id', 'label', 'pred_prob')
        train_metrics['pred_prob'].loc[training_data['cate']==key] = y_train_predict
        submit_ID['pred_prob'].loc[sub_trainning_date['cate']==key] = y_predict

    scores = 0
    train_metrics = train_metrics.groupby(['user_id', 'cate', 'shop_id'], as_index=False).sum()
    train_metrics = train_metrics.sort_values(by='pred_prob', ascending=False)
    # train_metrics.drop_duplicates(['user_id','cate','shop_id'])
    train_metrics = train_metrics.reset_index()
    size = train_metrics.shape[0]
    for i in [int(i*size*0.01-1) for i in range(1, 100)]:
        threshold = train_metrics['pred_prob'].iloc[i]
        scores_temp = get_metrice(train_metrics, threshold=threshold)
        print(str(i) + '  ' + str(scores_temp))
        if scores_temp > scores:
            scores = scores_temp
            fin_threshold = threshold
    print('inline score ' + str(scores) + 'threshold' + str(fin_threshold))

    submit_ID = submit_ID.groupby(['user_id', 'cate', 'shop_id'], as_index=False).sum()
    submit_ID = submit_ID.sort_values(by='pred_prob', ascending=False)
    submit_ID = submit_ID.reset_index()
    submit_ID.to_csv('score.csv', header=True, sep=',', index=0)
    submit_ID = submit_ID.loc[0:20000]
    submit_ID[['user_id', 'cate', 'shop_id']].astype(int).to_csv(
        '2019_5_19_lgbm_cate.csv', header=True, sep=',', encoding='utf-8', index=0)


if __name__ == '__main__':
    lgbm()
    # lgbm_cate()