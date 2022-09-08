from sklearn.linear_model import LogisticRegression
from evaluate_model import evaluate_model_performance_two_class, plot_roc
import numpy as np
from sklearn import preprocessing
from util import load_data
import pandas as pd
import random
from sklearn.model_selection import StratifiedShuffleSplit


def inter_set_LR(ratio):
    ratio = ratio
    path0 = "../data/source_data(huaxi)/featureV1.0/shuffle_feature"

    train_psd_features = np.load(path0 + '/patient_psd.npy', allow_pickle=True)
    train_de_features = np.load(path0 + '/patient_de.npy', allow_pickle=True)
    train_time_features = np.load(path0 + '/patient_time.npy', allow_pickle=True)
    train_age_and_sex = np.load(path0 + '/patient_age_and_sex.npy', allow_pickle=True)

    train_x = np.concatenate((train_psd_features.reshape(len(train_psd_features), -1),
                              train_de_features.reshape(len(train_de_features), -1),
                              train_time_features.reshape(len(train_time_features), -1),
                              train_age_and_sex.reshape(len(train_age_and_sex), -1)), axis=-1)
    train_y = np.load(path0 + '/patient_label.npy', allow_pickle=True).reshape(-1)

    # path1 = "../data/target_data(hangqi)/featureV1.0/target_shuffle_feature"
    #
    # test_psd_features = np.load(path1 + '/patient_psd.npy', allow_pickle=True)
    # test_de_features = np.load(path1 + '/patient_de.npy', allow_pickle=True)
    # test_time_features = np.load(path1 + '/patient_time.npy', allow_pickle=True)
    # test_age_and_sex = np.load(path1 + '/patient_age_and_sex.npy', allow_pickle=True)
    #
    # test_x = np.concatenate((test_psd_features.reshape(len(test_psd_features), -1),
    #                          test_de_features.reshape(len(test_de_features), -1),
    #                          test_time_features.reshape(len(test_time_features), -1),
    #                          test_age_and_sex.reshape(len(test_age_and_sex), -1)), axis=-1)
    # test_y = np.load(path1 + '/patient_label.npy', allow_pickle=True).reshape(-1)
    #
    # hc_data = test_x[test_y == 0]
    # hc_label = test_y[test_y == 0]
    # if ratio == 1:  # HC:SCZ = 1：1
    #     scz_data = test_x[test_y == 1][-hc_data.shape[0]:, :]
    #     scz_label = test_y[test_y == 1][-hc_data.shape[0]:]
    # elif ratio == 0.5:  # HC:SCZ = 1：2
    #     scz_data = test_x[test_y == 1][:2 * hc_data.shape[0], :]
    #     scz_label = test_y[test_y == 1][:2 * hc_data.shape[0]]
    # elif ratio == 0.3:  # HC:SCZ = 1：3
    #     scz_data = test_x[test_y == 1]
    #     scz_label = test_y[test_y == 1]
    # elif ratio == 2:  # HC:SCZ = 2：1
    #     scz_data = test_x[test_y == 1][:int(0.5 * hc_data.shape[0]), :]
    #     scz_label = test_y[test_y == 1][:int(0.5 * hc_data.shape[0])]
    # elif ratio == 3:  # HC:SCZ = 3：1
    #     scz_data = test_x[test_y == 1][:int(0.33 * hc_data.shape[0]), :]
    #     scz_label = test_y[test_y == 1][:int(0.33 * hc_data.shape[0])]
    # # print("Hangqi scz：", scz_data.shape[0])
    #
    # tst_data = np.concatenate((hc_data, scz_data), axis=0)
    # tst_label = np.concatenate((hc_label, scz_label), axis=0)
    # index = [i for i in range(len(tst_data))]
    # random.shuffle(index)
    # test_x = tst_data[index]
    # test_y = tst_label[index]
    # print('Hangqi: ', pd.value_counts(test_y))

    path1 = "../data/Warsaw/hc_clean/all_hc_feature"
    path2 = "../data/Warsaw/scz_clean/all_scz_feature"

    test_psd_features = np.concatenate((np.load(path1 + '/all_psd_feature.npy', allow_pickle=True),
                                        np.load(path2 + '/all_psd_feature.npy', allow_pickle=True)), axis=0)
    test_de_features = np.concatenate((np.load(path1 + '/all_de_feature.npy', allow_pickle=True),
                                       np.load(path2 + '/all_de_feature.npy', allow_pickle=True)), axis=0)
    test_time_features = np.concatenate((np.load(path1 + '/all_time_feature.npy', allow_pickle=True),
                                         np.load(path2 + '/all_time_feature.npy', allow_pickle=True)), axis=0)
    test_age_and_sex = np.concatenate((np.load(path1 + '/all_age_and_sex_feature.npy', allow_pickle=True),
                                       np.load(path2 + '/all_age_and_sex_feature.npy', allow_pickle=True)), axis=0)

    _test_x = np.concatenate((test_psd_features.reshape(len(test_psd_features), -1),
                             test_de_features.reshape(len(test_de_features), -1),
                             test_time_features.reshape(len(test_time_features), -1),
                             test_age_and_sex.reshape(len(test_age_and_sex), -1)), axis=-1)
    _test_y = np.concatenate((np.zeros(14), np.ones(14)), axis=0)

    index = [i for i in range(len(_test_x))]
    random.shuffle(index)
    test_x = _test_x[index]
    test_y = _test_y[index]
    print('russia set label: ', pd.value_counts(test_y))

    logistic_regression = LogisticRegression(max_iter=10000, solver='lbfgs')
    min_max_scaler = preprocessing.MinMaxScaler()
    train_x = min_max_scaler.fit_transform(train_x)
    logistic_regression.fit(train_x, train_y.reshape(-1, ))
    test_x = min_max_scaler.transform(test_x)
    predicted_y_prob = logistic_regression.predict_proba(test_x)[:, 1]
    acc, recall, precision, f1, specificity, auc = evaluate_model_performance_two_class(test_y,
                                                                                        predicted_y_prob,
                                                                                        flag=False)
    return acc, recall, precision, f1, specificity, auc

def intra_set_LR():
    path0 = "../data/source_data(huaxi)/featureV1.0/shuffle_feature"

    train_psd_features = np.load(path0 + '/patient_psd.npy', allow_pickle=True)
    train_de_features = np.load(path0 + '/patient_de.npy', allow_pickle=True)
    train_time_features = np.load(path0 + '/patient_time.npy', allow_pickle=True)
    train_age_and_sex = np.load(path0 + '/patient_age_and_sex.npy', allow_pickle=True)

    data = np.concatenate((train_psd_features.reshape(len(train_psd_features), -1),
                             train_de_features.reshape(len(train_de_features), -1),
                             train_time_features.reshape(len(train_time_features), -1),
                             train_age_and_sex.reshape(len(train_age_and_sex), -1)), axis=-1)
    label = np.load(path0 + '/patient_label.npy', allow_pickle=True).reshape(-1)

    all_acc = []
    all_recall = []
    all_precision =[]
    all_f1 = []
    all_specificity =[]
    all_auc = []
    k_fold = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
    for trn_idex, tst_idex in k_fold.split(data, label):
        train_x = data[trn_idex]
        test_x = data[tst_idex]
        train_y = label[trn_idex]
        test_y = label[tst_idex]

        logistic_regression = LogisticRegression(max_iter=10000, solver='lbfgs')
        min_max_scaler = preprocessing.MinMaxScaler()
        train_x = min_max_scaler.fit_transform(train_x)
        logistic_regression.fit(train_x, train_y.reshape(-1, ))
        test_x = min_max_scaler.transform(test_x)
        predicted_y_prob = logistic_regression.predict_proba(test_x)[:, 1]
        acc, recall, precision, f1, specificity, auc = evaluate_model_performance_two_class(test_y,
                                                                                            predicted_y_prob,
                                                                                            flag=False)

        all_acc.append(acc)
        all_recall.append(recall)
        all_precision.append(precision)
        all_f1.append(f1)
        all_specificity.append(specificity)
        all_auc.append(auc)

    return np.mean(all_acc), np.mean(all_recall), np.mean(all_precision), np.mean(all_f1), np.mean(all_specificity), np.mean(all_auc)


if __name__ == '__main__':
    ratio = 1

    all_acc = []
    all_recall = []
    all_precision = []
    all_f1 = []
    all_specificity = []
    all_auc = []
    for i in range(500):
        acc, recall, precision, f1, specificity, auc = inter_set_LR(ratio)
        # acc, recall, precision, f1, specificity, auc = intra_set_LR()
        all_acc.append(acc)
        all_recall.append(recall)
        all_precision.append(precision)
        all_f1.append(f1)
        all_specificity.append(specificity)
        all_auc.append(auc)
    print('final_result: auc--{:.3f}±{:.3f}-acc--{:.3f}±{:.3f}--precision--{:.3f}±{:.3f}-'
          '-recall--{:.3f}±{:.3f}--f1--{:.3f}±{:.3f}-'
          '-specificity--{:.3f}±{:.3f}'.format(np.mean(all_auc),
                                               np.std(all_auc),
                                               np.mean(all_acc),
                                               np.std(all_acc),
                                               np.mean(all_precision),
                                               np.std(all_precision),
                                               np.mean(all_recall),
                                               np.std(all_recall),
                                               np.mean(all_f1),
                                               np.std(all_f1),
                                               np.mean(all_specificity),
                                               np.std(all_specificity)))