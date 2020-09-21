# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 17:31:26 2019

"""

import datetime
import pandas as pd
import numpy as np
import BestClassifierSearch as bc
# import lime
import Time_Functions as tf
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

np.set_printoptions(suppress=True)
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, accuracy_score, classification_report,     confusion_matrix, precision_recall_curve, auc, make_scorer, recall_score, precision_score
from imblearn.over_sampling import SMOTE
from collections import Counter
import Generic_Functions as gf
# from lime.lime_text import LimeTextExplainer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.multiclass import OneVsRestClassifier
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt


def run_model(clf_v, extra_param=None, v_list_general=None, v_n_jobs = None):
    #    global params

    if type(clf_v).__name__ in ["DecisionTreeClassifier"]:
        param_grid_name = "param_grid_" + "TREE"
    else:
        param_grid_name = "param_grid_" + type(clf_v).__name__
    if extra_param is None:
        model_execute_string = '''params = bc.gridsearchCLF(''' + param_grid_name + ''' ,clf_v  ,type(clf_v).__name__ ,X_train, y_train, X_test,  y_test ,v_n_jobs, "xxx" ,v_cv = 5)'''
    else:
        model_execute_string = '''params = bc.gridsearchCLF(''' + param_grid_name + ''' ,clf_v  ,type(clf_v).__name__ ,X_train, y_train, X_test,  y_test ,v_n_jobs, extra_param , v_list_general ,v_cv = 5)'''
    print(model_execute_string)
    exec(model_execute_string)
    # return params


# For detailed information behind the interpretability of ROC is explained in detailed from following:
# https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5
def ROCGraph(clf_v, X_test, y_test, title_v):
    y_pred_prob = clf_v.predict_proba(X_test)
    # if it is a binary classification, then 1 value is reference point
    # but if you have multiclass problem ROC curve becomes one vs. ALL graph.
    # one has to change y_pred_prob[:,1] according to reference class
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob[:, 1])
    plt.plot(fpr, tpr)
    plt.xlim([-0.05, 1.1])  ##just for seeing line in case of perferct classifier alligns with Y axis
    plt.ylim([-0.05, 1.1])
    plt.rcParams['font.size'] = 12
    plt.title('ROC curve for ' + title_v)
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.grid(True)
    plt.show()
    auc = roc_auc_score(y_test, y_pred_prob[:, 1])
    print("")
    print('AUC: %.3f' % auc)


def find_similar_instance(v_ref_index, v_X, v_y, v_df_final, v_actual_label):
    '''
    :param v_ref_index:  which instance you have to assign as reference instance
    :param v_X: X of original model
    :param v_y: y of original model
    :param v_df_final: dataframe which is similar index as X, and includes TEXT column
    :param v_actual_label: which actual instance you want to investigate
    :return:
    '''
    v_query = "Label==" + str(v_actual_label)
    ref_data = pd.concat([v_X.loc[[v_ref_index]] ,  v_X.join(v_y).query(v_query) [v_X.columns] ])
    tst_similarity = gf.f_create_similarity(ref_data,
                                            v_ref_index,
                                            v_numeric_columns_list=v_X.columns,
                                            v_categoric_column_list=[])
    similar_instances = tst_similarity[(tst_similarity.compare_index != v_ref_index) & (tst_similarity.tot_distance < 0.2)]
    if len(similar_instances) == 0:
        return '***'
    else:
        return '***' + similar_instances.head(1).compare_index.values[0] + '***' + '\n' + v_df_final.loc[similar_instances.head(1).compare_index, 'TEXT'].values[0]



def f_create_test_table(predictor, X_test, y_test,  v_actual_label, v_predict_label,
                        thr_value=0.5,
                        classes_map = None,
                        orig_table = None,
                        v_output_file_name = None,
                        v_similarity_source = None # if we need to add similar samples to excel, it should be filled by X,y
                        ):
    predictions = predictor.predict_proba(X_test)
    if classes_map is not None:
        y_test = y_test.map(classes_map)
        final_table = pd.DataFrame(data=predictions, columns=[str(x) + "_prob" for x in [classes_map.get(x) for x in predictor.classes_]],
                               index=y_test.index)
    else:
        final_table = pd.DataFrame(data=predictions, columns=[str(x) + "_prob" for x in predictor.classes_],
                               index=y_test.index)
    final_table["final_prob"] = final_table.apply(np.max, axis=1)
    final_table["PRED"] = final_table.idxmax(axis=1).apply(lambda x: x[0])
    final_table["ACTUAL"] = y_test.values
    final_table["FINAL_PRED"] = final_table.apply(
        lambda row: int(-99) if row["final_prob"] < thr_value else int(row["PRED"]), axis=1)
    part_len = len(final_table[final_table["FINAL_PRED"] == -99])
    full_len = len(final_table)
    print(part_len)
    print(full_len)
    print("%" + str(float(part_len) / full_len * 100) + " is lossed")

    print(classification_report(final_table[final_table["FINAL_PRED"] != -99]["ACTUAL"],
                                final_table[final_table["FINAL_PRED"] != -99]["FINAL_PRED"]))
    print(confusion_matrix(final_table[final_table["FINAL_PRED"] != -99]["ACTUAL"],
                           final_table[final_table["FINAL_PRED"] != -99]["FINAL_PRED"]))

    ##round decimal values to 3
    final_table[final_table.columns[final_table.dtypes == "float64"]] = final_table[final_table.columns[final_table.dtypes == "float64"]].apply(lambda x: np.round(x, 3), axis=1)

    dummy_list = []

    if classes_map is not None:
        predictor_classes_ = [classes_map.get(x) for x in predictor.classes_]
    else:
        predictor_classes_ = predictor.classes_

    for v_target in predictor_classes_:
        upper_thr = 1.0
        lower_thr = 1.0
        while lower_thr > 0:
            lower_thr = np.round(upper_thr - 0.1, 4)
            dummy_Table = final_table[(final_table[str(v_target) + "_prob"] > lower_thr) & (
                        final_table[str(v_target) + "_prob"] <= upper_thr)]
            num_of_total = len(dummy_Table)
            num_of_correct = len(dummy_Table[dummy_Table["ACTUAL"] == v_target])
            if num_of_total == 0:
                correct_ratio = 0
            else:
                correct_ratio = np.round(float(num_of_correct) / num_of_total, 4)

            dummy_list.append(
                [v_target, str(np.round(lower_thr, 4)) + "-" + str(upper_thr), num_of_total, num_of_correct,
                 correct_ratio])
            upper_thr = lower_thr


    #####Create test table which shows wrong predicted values####
    if v_output_file_name is None:
        file_name_html = "NLP_Wrong_Results.html"
        file_name_xls = "NLP_Wrong_Results.xlsx"
    else:
        file_name_html = v_output_file_name + ".html"
        file_name_xls = v_output_file_name + ".xlsx"

    if orig_table is not None:
        if "TEXT" in orig_table.columns:
            column_name = "TEXT"
        else:
            column_name = "Text"
        column_name_norm = column_name + "_norm"
        query_string = '''(FINAL_PRED != "-99")''' + " & "  + \
                       '(' + '''ACTUAL ==''' + '''"''' + str(v_actual_label) +  '''"''' + ' & ' + 'PRED ==' + '''"''' + str(v_predict_label) + '''"''' + ')'
        print(query_string)
        if "selected_features" in final_table.columns:
            incost_df = final_table.query(query_string).join(orig_table[[column_name, column_name_norm, "selected_features"]])
            if v_similarity_source is not None:
                incost_df.reset_index(inplace=True)
                incost_df["similar_instance"] = incost_df.ID.apply(
                    lambda x: find_similar_instance(x, v_similarity_source[0], v_similarity_source[1], orig_table,
                                                    v_predict_label))
                incost_df.set_index("ID", inplace=True)
            incost_df.to_html(file_name_html)
            incost_df.to_excel(file_name_xls)
            # final_table.query('(FINAL_PRED != "-99") & (PRED == "2" & ACTUAL == "1")').join(orig_table[[column_name, column_name_norm, "selected_features"]]).to_html(file_name_html)
            # final_table.query('(FINAL_PRED != "-99") & (PRED == "2" & ACTUAL == "1")').join(orig_table[[column_name, column_name_norm, "selected_features"]]).to_excel(file_name_xls)
        else:
            incost_df = final_table.query(query_string).join(orig_table[[column_name, column_name_norm]])
            if v_similarity_source is not None:
                incost_df.reset_index(inplace=True)
                incost_df["similar_instance"] = incost_df.ID.apply(
                    lambda x: find_similar_instance(x, v_similarity_source[0], v_similarity_source[1], orig_table,
                                                    v_predict_label))
                incost_df.set_index("ID", inplace=True)
            incost_df.to_html(file_name_html)
            incost_df.to_excel(file_name_xls)
            # final_table.query('(FINAL_PRED != "-99") & (PRED == "2" & ACTUAL == "1")').join(orig_table[[column_name, column_name_norm]]).to_html(file_name_html)
            # final_table.query('(FINAL_PRED != "-99") & (PRED == "2" & ACTUAL == "1")').join(orig_table[[column_name, column_name_norm]]).to_excel(file_name_xls)

            # find_similar_instance(v_ref_index, v_X, v_y, v_df_final, v_actual_label):
    return final_table, pd.DataFrame(data=dummy_list,
                                     columns=["target", "prob_bin", "total_samples", "hit_samples", "ratio"])


#########################################GENERIC Functions Finish######################################


#######################PARAMS DEFINITIONS FOR HYPERPARAMETER TUNING######################
param_grid_TREE = {
    # splitting criterion is just for directing model to choose correct metric.
    # General possibilities are: gini, entropy, chi square
    # All of them has tendency to split node to significant differentiate of target variable.
    # For example you have 100sammples ant target variable YES (50 samples) and NO (50 samples). If your splitting
    # variable generates two sub nodes of which has 50% probabilty of having YES and NO meaning shit.
    # It should significantly differentiate YES ond NO, for instance 10%YES and %90 No for a subnode is better than it.
    # Useful detailed info is at:
    # https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/
    # One more note: if it is regression tree, i.e target is continous, splitting critterion is:
    # lower variance in splitted sub nodes. Because if variance between actual and predicted becomes
    # decreased compared the parent, it means classifier is on the way that correctly differentiated
    # target
    # 'criterion': ['gini', 'entropy'],
    #    'min_samples_split': [ 3, 5, 8],
    # 'max_depth': [8, 10, 15, 20],
    'max_depth': [10],
    #    'min_samples_leaf': [ 10, 12],
    'random_state': [0]
    , 'class_weight': ["balanced"]
}

param_grid_RandomForestClassifier = {
    #    'min_samples_split': [2, 3, 5, 8],
    # 'max_depth': [30 ,40],
    'max_depth': [10],
    'random_state': [137],
    'class_weight': ["balanced"],
    'n_estimators': [150, 200]
}

# Note that SVM is so time consuming especially kernels different than linear
param_grid_SVC = {
    # 'kernel': ('linear', 'rbf'),
    'kernel': ['linear'],
    # 'kernel':['linear'],
    # 'C':(1,0.25,0.5,0.75), # C is just for regularization. There is a tradeoff between train error
    #                        # and the largest minimum margin of hyperplane (test set error which model does not see yet)
    #                        # for detailed info, following site is so helpful: https://stats.stackexchange.com/questions/31066/what-is-the-influence-of-c-in-svms-with-linear-kernel
    # 'C': [0.1, 0.01, 1],  # C is just for regularization. There is a tradeoff between train error
    'C': [1],  # C is just for regularization. There is a tradeoff between train error
    # and the largest minimum margin of hyperplane (test set error which model does not see yet)
    # for detailed info, following site is so helpful: https://stats.stackexchange.com/questions/31066/what-is-the-influence-of-c-in-svms-with-linear-kernel
    # 'gamma': (1,2,3,'auto'),
    'gamma': ['auto'],
    # When gamma is low, the ‘curve’ of the decision boundary is very low and thus the decision region is very broad. When gamma is high, the ‘curve’ of the decision boundary is high, which creates islands of decision-boundaries around data points.
    # 'decision_function_shape':['ovr'], # ovo: one vs. one : Assume that you have 3 classes A,B,C. OVO assigns classifiers for a sample being A or B, A or C, B or C
    # ovr: one vs. rest : Assume that you have 3 classes A,B,C. OVO assigns classifiers for a sample being A or Not, B or NOT, C or NOT
    # https://scikit-learn.org/dev/modules/multiclass.html
    'shrinking': [True]
    # 'shrinking':[True]
}

# because NB has not any params
param_grid_MultinomialNB = {
    'alpha': [0.001, 0.01, 0.1, 0.5, 0.7, 1]
}

# https://machinelearningmastery.com/start-with-gradient-boosting/
# For Gradient  Boosting Algorithm
param_grid_GradientBoostingClassifier = {  # 'learning_rate': [0.1, 0.05, 0.02, 0.01],
    'learning_rate': [0.1, 1, 10],
    'n_estimators': [100, 150],
    'random_state': [0]
    # 'max_features': [1.0, 0.3, 0.1]
}

# For Gradient  Boosting Algorithm
param_grid_AdaBoostClassifier = {  # 'learning_rate': [0.1, 0.05, 0.02, 0.01],
    'learning_rate': [0.1, 0.5, 1],
    'n_estimators': [100, 150],
    'random_state': [137]
    # 'max_features': [1.0, 0.3, 0.1]
}

param_grid_MLPClassifier = {
    #                'activation' :  ['identity', 'logistic', 'tanh', 'relu'],
    'solver': [ 'adam']
    # ,'solver': ['lbfgs', 'adam']
    #                'alpha': [0.001, 0.7],
    ,'batch_size': ['auto'],
    #                'batch_size': [20, 30, 40, 50],
    'hidden_layer_sizes': [(3,), (4,), (5,)]
    ,'max_iter': [500]
    #                'max_iter': [200],
    #                'tol' : [0.0001],
    ,'verbose': [True]
    # ,'warm_start': [False, True]
}

# For Gradient  KNN
param_grid_KNeighborsClassifier = {
    'algorithm': ['auto'],
    # {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’} how to find closest N samples. Best way is leaving AUTO
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19],  # most significant param.
    'weights': ['uniform']
    # uniform (each sample has equal weight) vs. distance (if a sample is closer to test point than another, it has more weight)
}

# For Logistic Regression
# detailed info: https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_sample_weight.html
param_grid_LogisticRegression = {
    'C': [0.1, 1, 10, 100]
    # regularization parm: there is a tradeoff between minimizing training error and test set accuracy. It can thought as similar with SVM regularization param
    # ,'class_weight' : ['balanced', None] #“balanced”, or None, optional. default is NONE which causes all class has equal weight. BALANCED leads minor class samples have more weight (might be thought as oversampling for minority classes)
    , "penalty": ["l2"]
    #                ,"penalty": ["l2", "elasticnet"]
    #                ,"penalty": ["l2"]
    , 'random_state': [137]
    #                ,"solver" : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    #                ,"solver" : ['saga']
}


#######################PARAMS DEFINITIONS FOR HYPERPARAMETER TUNING FINISH######################

#######TRAIN-TEST Split###########
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=4234)
X_columns = X_train.columns

#######TRAIN-TEST Split Finish###########


##########OVERSAMPLING IF THERE IS ANY IMBALANCED CLASSES#########
sm = SMOTE(random_state=45)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())
X_train = X_train_res
y_train = y_train_res

gf.f_pickle_dump("x_y_train_tr.pkl", (X_train, X_test, y_train, y_test))
(X_train, X_test, y_train, y_test) = gf.f_pickle_load("x_y_train_tr.pkl")


##########Classifiers Assignment#########
clf_v = MultinomialNB()
clf_v = DecisionTreeClassifier()
clf_v = LogisticRegression()
clf_v = RandomForestClassifier()
clf_v = SVC(probability=True)
clf_v = GradientBoostingClassifier()
clf_v = AdaBoostClassifier()
clf_v = KNeighborsClassifier()
clf_v = MLPClassifier()



#############RUN MODEL WITH KBEST AND HYPERPARAMETER TUNING##############
# list_general = gf.f_pickle_load("list_general_model_kpi.pkl")
(X_train, X_test, y_train, y_test) = gf.f_pickle_load("x_y_train_tr.pkl")
X_train_c = X_train.copy()
X_test_c = X_test.copy()
list_general = gf.f_pickle_load("list_general_model_kpi.pkl")
global list_general
# list_general = []
####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
LANG = "adfasdasdasdas"  ####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# for x in range (600, X_train_c.shape[1], 100):
# for x in range (600, 901, 150):
for x in range(1600, 2400, 200):
    TOP_K = x
    selector = SelectKBest(f_classif, k=min(TOP_K, X_train_c.shape[1]))
    selector.fit(X_train_c, y_train)
    X_train = selector.transform(X_train_c)
    X_test = selector.transform(X_test_c)

    #  ######If need Normalization ######
    # from sklearn import preprocessing
    # mm_scaler = preprocessing.StandardScaler()
    # X_train = mm_scaler.fit_transform(X_train)
    # X_test = mm_scaler.transform(X_test)
    #  ##################################

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    print("$$$$$$$$  " + str(x) + "   $$$$$$$$")
    v_n_jobs = 1
    run_model(clf_v, extra_param=[x, LANG, type(clf_v).__name__,
                                           tf.formatDate(datetime.datetime.now(), "%Y-%m-%d %H:%M:%S")],
                       v_list_general=list_general,
                       v_n_jobs = -1)
    print("$$$$$$$$$$$$$$$$$$$$$")
    gf.f_pickle_dump("list_general_model_kpi.pkl"  ,list_general)
#############RUN MODEL WITH KBEST AND HYPERPARAMETER TUNING##############


#################Eski listeyi tekrar yüklemek istersek###############
with open("list_general_nlp_scores.pkl", 'rb') as f:
    list_general = pickle.load(f, encoding="latin1")
#############Eski listeyi tekrar yüklemek istersek FINISH############


# gf.f_pickle_dump("list_general_model_kpi.pkl", list_general)
list_general = gf.f_pickle_load("list_general_model_kpi.pkl")

for x in list_general:
    if x[0][2] == "LogisticRegression":
        print("**********************")
        print(x[0])
        print(x[1])
        print("$$$TEST KPI$$$")
        print(x[2])
        print(x[3])
        print("$$$TRAIN KPI$$$")
        print(x[4])
        print(x[5])
        print("**********************")


##########Classifiers Assignment#########
clf_v = MultinomialNB()
clf_v = DecisionTreeClassifier()
clf_v = LogisticRegression()
clf_v = RandomForestClassifier()
clf_v = SVC()
clf_v = GradientBoostingClassifier()
clf_v = AdaBoostClassifier()
clf_v = KNeighborsClassifier()
clf_v = MLPClassifier()

import warnings
warnings.filterwarnings("ignore")
#######################RUN MODEL WITHOUT KBEST AND BUT HYPERPARAMETER TUNING##########################
global list_general
list_general = []
####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
LANG = "TR"  ####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

run_model(clf_v, extra_param=['9999', LANG, type(clf_v).__name__,
                                       tf.formatDate(datetime.datetime.now(), "%Y-%m-%d %H:%M:%S")],
                   v_list_general=list_general,
                   v_n_jobs = -1)
print("$$$$$$$$$$$$$$$$$$$$$")

#######################RUN MODEL WITHOUT KBEST AND HYPERPARAMETER TUNING FINISH##########################

# for a in [[x[0], x[2]] for x in list_general_copy if x[0] == 900]:
# for a in [x for x in list_general_copy if x[0] == 900]:
# dummy_list = [x for x in [x for x in list_general if x[0][2] == "LogisticRegression" if x[0][1] == "TR"]]
dummy_list = [x for x in [x for x in list_general if x[0][2] == "RandomForestClassifier" ]]
i = 0
for a in dummy_list:
    # print(i)
    # i += 1
    # if i < 10 or i > 20:
    #     continue
    print(a[0][0])
    print(a[1])
    print(a[2])
    print(a[3])
    print(a[4])
    print(a[5])

    #    print(a[2])
    #    print(a[3])
    #    print(a[4])
    #    print(a[5])
    print("*******")


#######################RUN MODEL WITH Just KBEST##########################

predictor = SVC(probability=True)
params = {'C': 1, 'gamma': 'auto', 'kernel': 'linear', 'shrinking': True}

predictor = RandomForestClassifier()
params = {'max_depth': 20, 'n_estimators': 200, 'random_state': 137}

predictor = LogisticRegression()
params = {'C': 100, 'penalty': 'l2', 'random_state': 137}

predictor = MLPClassifier()
params = {'batch_size': 'auto', 'hidden_layer_sizes': (5,), 'max_iter': 500, 'solver': 'adam', 'verbose': True, 'random_state' : 137 }

predictor = GradientBoostingClassifier()
params = {'learning_rate': 1, 'n_estimators': 150, 'random_state': 0}

predictor = AdaBoostClassifier()
params = {'learning_rate': 1, 'n_estimators': 150, 'random_state': 137}

# params = {'batch_size': ['auto'], 'hidden_layer_sizes': [(4,)], 'solver': ['adam'], 'verbose': [False], 'warm_start': [True]}
# params = {'max_depth': 50, 'n_estimators': 150, 'random_state': 137}
# params = {'penalty': 'l2', 'C': 10, 'random_state': 137}
predictor.set_params(**params)
selector = SelectKBest(f_classif, k=2200)
selector.fit(X_train, y_train)
# selector.fit(X_train_c, y_train)
# X_train = selector.transform(X_train_c)
# X_test = selector.transform(X_test_c)
X_train = selector.transform(X_train)
X_test = selector.transform(X_test)
######If need Normalization ######
# from sklearn import preprocessing
# mm_scaler = preprocessing.StandardScaler()
# X_train = mm_scaler.fit_transform(X_train)
# X_test = mm_scaler.transform(X_test)
##################################
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

predictor.fit(X_train, y_train)

# save it below
model_name = "predictor_SVC_tr_model.pkl"
gf.f_pickle_dump(model_name, (predictor, selector, Tfidf_Vector, X_test, y_test, X_columns)  )
(predictor, selector, Tfidf_Vector, X_test, y_test, X_columns) = gf.f_pickle_load(model_name)
# clf.fit(X_train, y_train)
#######################RUN MODEL WITH Just KBEST##########################


#######################JUST RUN MODEL##########################
predictor = DecisionTreeClassifier()
predictor = RandomForestClassifier()
predictor = LogisticRegression()
predictor = MLPClassifier()
predictor = SVC(probability=True)
from xgboost import XGBClassifier
predictor = XGBClassifier()

# predictor = LogisticRegression()
params = {'C': 100, 'penalty': 'l2', 'random_state': 137} ##
params = {'max_depth': 10, 'n_estimators': 150, 'random_state': 137}
params = {'class_weight': 'balanced', 'max_depth': 10, 'random_state': 0}
params = {'batch_size': 'auto', 'hidden_layer_sizes': (5,), 'max_iter': 300, 'solver': 'adam', 'verbose': True}
params = {'kernel': 'linear', 'C': 1, 'gamma': 'auto', 'shrinking': True}

# params = {'penalty': 'l2', 'C': 10, 'random_state': 137}
predictor.set_params(**params)
predictor.fit(X_train, y_train)

#######################JUST RUN MODEL FINISH##########################



test_table, lift_table = f_create_test_table(predictor, X_test, y_test, thr_value=0.7, classes_map = None, orig_table = df_final_text, v_output_file_name = "ITSM_2ACTUAL_1PREDICT")

test_table, lift_table = f_create_test_table(predictor, X_test, y_test, thr_value=0.8, classes_map = None, orig_table = df_final, v_output_file_name = "Predict2_Actual1_stack_trace")

# exclude_list = []
test_table, lift_table = f_create_test_table(predictor,
                                             X_test.loc[[x for x in X_test.index if x not in exclude_list]],
                                             y_test.loc[[x for x in X_test.index if x not in exclude_list]],
                                             v_actual_label = 2,
                                             v_predict_label = 1,
                                             # X_test,
                                             # y_test,
                                             thr_value=0.75,
                                             classes_map = None,
                                             orig_table = df_final,
                                             v_output_file_name = "_Predict1Actaul2_",
                                             # v_similarity_source = [X,y]  # None to not create similarity matrix,  to do it use: [X,y]
                                             v_similarity_source = None  # None to not create similarity matrix,  to do it use: [X,y]
                                             )

