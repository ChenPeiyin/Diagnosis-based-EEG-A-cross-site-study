from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, accuracy_score, roc_curve, auc, confusion_matrix
import numpy as np
from sklearn import metrics
import xlwt
import matplotlib.pyplot as plt
from sklearn import preprocessing
import shutil
import matplotlib
shutil.rmtree(matplotlib.get_cachedir())


def evaluate_model_performance(labels, one_hot_labels, predicted_labels_prob, predicted, flag=True):
    labels = labels.reshape(-1,)
    aucs = []
    specificity = []

    for i in range(4):
        label = one_hot_labels[:, i]
        predicted_each_label_prob = predicted_labels_prob[:, i]

        fpr, tpr, thresholds = roc_curve(label, predicted_each_label_prob, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        aucs.append(auc)

    accuracy = accuracy_score(labels, predicted)

    cfmetric = confusion_matrix(labels, predicted, labels=[1, 0])
    tn, fp, fn, tp = cfmetric.ravel()
    tnr = tn / (tn + fp)
    specificity.append(tnr)

    macro_auc = roc_auc_score(labels, predicted_labels_prob, average='macro', multi_class='ovo')
    macro_precision = precision_score(labels, predicted, average='macro')
    macro_recall = recall_score(labels, predicted, average='macro')
    macro_f1 = f1_score(labels, predicted, average='macro')



    # micro_auc = roc_auc_score(labels, predicted_labels_prob, average='micro', multi_class='ovo')
    micro_precision = precision_score(labels, predicted, average='micro')
    micro_recall = recall_score(labels, predicted, average='micro')
    micro_f1 = f1_score(labels, predicted, average='micro')

    weighted_auc = roc_auc_score(labels, predicted_labels_prob, average='weighted', multi_class='ovo')
    weighted_precision = precision_score(labels, predicted, average='weighted')
    weighted_recall = recall_score(labels, predicted, average='weighted')
    weighted_f1 = f1_score(labels, predicted, average='weighted')
    if flag is True:
        print('mean_auc--{}, macro_auc--{}, accuracy--{}, macro_precision--{}, macro_recall--{}, macro_f1--{},'
              'micro_precision--{}, micro_recall--{}, micro_f1--{},'
              'weighted_auc--{}, weighted_precision--{}, weighted_recall--{}, weighted_f1--{}'.
              format(np.mean(aucs), macro_auc, accuracy, macro_precision, macro_recall, macro_f1,
                     micro_precision, micro_recall, micro_f1,
                     weighted_auc, weighted_precision, weighted_recall, weighted_f1))
    return macro_auc, accuracy, macro_precision, macro_recall, macro_f1


def evaluate_model_performance_two_class(labels, predicted_labels_prob, flag=True):
    labels = labels.reshape(-1,)
    predicted_labels_prob = predicted_labels_prob.reshape(-1,)
    fpr, tpr, thresholds = roc_curve(labels, predicted_labels_prob, pos_label=1)
    threshold = thresholds[np.argmax(tpr - fpr)]
    # threshold = 0.63
    auc = roc_auc_score(labels, predicted_labels_prob)
    y_pred_labels = (predicted_labels_prob >= threshold) * 1
    cm = confusion_matrix(labels, y_pred_labels)
    tn, fp, fn, tp = confusion_matrix(labels, y_pred_labels).ravel()

    acc = accuracy_score(labels, y_pred_labels)
    recall = recall_score(labels, y_pred_labels)
    precision = precision_score(labels, y_pred_labels)
    f1 = f1_score(labels, y_pred_labels)

    specificity = tn / (tn + fp)
    if flag is True:
        print('auc--{}--acc--{}--precision--{}--recall--{}--f1--{}--specificity--{}'.format(auc, acc,
                                                                                            precision,
                                                                                            recall,
                                                                                            f1,
                                                                                            specificity))
    return acc, recall, precision, f1, specificity, auc


def evaluate_model_performance_voting(labels, subject_labels, predicted_labels_prob):
    labels = labels.reshape(-1, )
    predicted_labels_prob = predicted_labels_prob.reshape(-1, )
    fpr, tpr, thresholds = roc_curve(labels, predicted_labels_prob, pos_label=1)
    threshold = thresholds[np.argmax(tpr - fpr)]
    y_pred_labels = (predicted_labels_prob >= threshold) * 1
    y_pred_labels = y_pred_labels.reshape(-1, 12)

    subject_predicted_labels = np.zeros(shape=[len(subject_labels)])
    for subject_index in range(len(subject_predicted_labels)):
        subject_predicted_labels[subject_index] = 1 if sum(y_pred_labels[subject_index]) >= 6 else 0
    evaluate_model_performance_two_class(subject_labels, subject_predicted_labels, True)


def plot_roc(test_labels, test_predictions, file_name):
    wb = xlwt.Workbook(file_name + ".xls")
    table = wb.add_sheet('Sheet1')
    table_title = ["test_index", "label", "prob", "pre", " ", "fpr", "tpr", "thresholds", " ", "fp", "tp", "fn", "tn",
                   "fp_words", "fp_freq", "tp_words", "tp_freq", "fn_words", "fn_freq", "tn_words", "tn_freq", " ",
                   "acc", "auc", "recall", "precision", "f1-score", "threshold"]
    fpr, tpr, thresholds = roc_curve(test_labels, test_predictions, pos_label=1)
    threshold = thresholds[np.argmax(tpr - fpr)]
    print(threshold)
    for i in range(len(fpr)):
        table.write(i + 1, table_title.index("tpr"), tpr[i])
        table.write(i + 1, table_title.index("fpr"), fpr[i])
        table.write(i + 1, table_title.index("thresholds"), float(thresholds[i]))
    table.write(2, table_title.index("threshold"), float(threshold))
    auc = "%.3f" % metrics.auc(fpr, tpr)
    title = file_name
    with plt.style.context('ggplot'):
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, "#000099", label='AUC = ' + str(auc))
        ax.plot([0, 1], [0, 1], 'k--', label='Baseline')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.title(title)
        plt.savefig(file_name + '.png', format='png')
        plt.close()
    return threshold


def normalize(features):
    val0 = np.mean(features[:, :, 0])
    val1 = np.mean(features[:, :, 1])
    val2 = np.mean(features[:, :, 2])
    val3 = np.mean(features[:, :, 3])
    val4 = np.mean(features[:, :, 4])

    features[:, :, 0] = features[:, :, 0] - val0
    features[:, :, 1] = features[:, :, 1] - val1
    features[:, :, 2] = features[:, :, 2] - val2
    features[:, :, 3] = features[:, :, 3] - val3
    features[:, :, 4] = features[:, :, 4] - val4

    features[:, :, 0] = 2 * features[:, :, 0] / val0
    features[:, :, 1] = 2 * features[:, :, 1] / val1
    features[:, :, 2] = 2 * features[:, :, 2] / val2
    features[:, :, 3] = 2 * features[:, :, 3] / val3
    features[:, :, 4] = 2 * features[:, :, 4] / val4

    features = features.reshape(-1, 16 * 5)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    features = min_max_scaler.fit_transform(features)
    return features.reshape(len(features), 16, 5)
