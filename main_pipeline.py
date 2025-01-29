import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import class_weight
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from scipy import stats
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
import matplotlib
import matplotlib.pyplot as plt

from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering
from catboost import CatBoostClassifier

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

def plot_importance(model, features, num, save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def stuttering_data_prep(df):

    df.rename(columns={"stutering": "stuttering"}, inplace=True)

    #mfcc
    mfcc_mean_columns = [col for col in df.columns if col.startswith("mfcc_") & col.endswith("mean")]
    df["mfcc_mean_sum"] = df[mfcc_mean_columns].sum(axis=1)

    mfcc_std_columns = [col for col in df.columns if col.startswith("mfcc_") & col.endswith("std")]
    df["mfcc_std_sum"] = df[mfcc_std_columns].sum(axis=1)

    mfcc_columns = [col for col in df.columns if col.startswith("mfcc_") & col.endswith(("_mean", "_std"))]

    df["mfcc_sum"] = df[mfcc_columns].sum(axis=1)


    #chroma
    chroma_mean_columns = [col for col in df.columns if col.startswith("chroma_") & col.endswith("mean")]
    df["chroma_mean_sum"] = df[chroma_mean_columns].sum(axis=1)

    chroma_std_columns = [col for col in df.columns if col.startswith("chroma_") & col.endswith("std")]
    df["chroma_std_sum"] = df[chroma_std_columns].sum(axis=1)

    df["rms_zcr_mean"] = (df["rms_mean"] + df["zcr_mean"])/2

    df["sc_sr_mean"] = (df["spectral_centroid_mean"] + df["spectral_rolloff_mean"])/2

    df["rms_std*sc_std_"] = df["rms_std"] * df["spectral_centroid_std"]

    df["tempo_+_zcr"] = df["tempo"] + df["zcr_mean"]

    df["mfcc_*_rms_mean"] = (df["mfcc_mean_sum"] * df["rms_mean"])/13

    df["chroma+rolloff"] = df[chroma_mean_columns].sum(axis=1) + df["spectral_rolloff_mean"]

    df["tempo_*_zcr"] = df["tempo"] + df["rms_mean"]

    df["sc_sr_chroma_mean"] = df["spectral_centroid_mean"] + df["spectral_rolloff_mean"] + df["chroma_mean_sum"]

    #############Hale
    df["mfcc_mean_avg"] = (df["mfcc_1_mean"] + df["mfcc_2_mean"] + df["mfcc_3_mean"]+ df["mfcc_4_mean"]+ df["mfcc_5_mean"]+ df["mfcc_6_mean"]+ df["mfcc_7_mean"]+ df["mfcc_8_mean"]+ df["mfcc_9_mean"]+ df["mfcc_10_mean"]+ df["mfcc_11_mean"]+ df["mfcc_12_mean"]+ df["mfcc_13_mean"]) / 13

    df["mfcc_mean_var"] = (df["mfcc_1_std"] + df["mfcc_2_std"] + df["mfcc_3_std"] + df["mfcc_4_std"] + df["mfcc_5_std"] + df["mfcc_6_std"] + df["mfcc_7_std"] + df["mfcc_8_std"] + df["mfcc_9_std"] + df["mfcc_10_std"] + df["mfcc_11_std"] + df["mfcc_12_std"] + df["mfcc_13_std"]) / 13

    df["mfcc_2_1"] = df["mfcc_2_mean"] - df["mfcc_1_mean"]

    df["mfcc_3_2"] = df["mfcc_3_mean"] - df["mfcc_2_mean"]

    df["mfcc_4_3"] = df["mfcc_4_mean"] - df["mfcc_3_mean"]

    df["mfcc_5_4"] = df["mfcc_5_mean"] - df["mfcc_4_mean"]

    df["mfcc_6_5"] = df["mfcc_6_mean"] - df["mfcc_5_mean"]

    df["mfcc_7_6"] = df["mfcc_7_mean"] - df["mfcc_6_mean"]

    df["mfcc_8_7"] = df["mfcc_8_mean"] - df["mfcc_7_mean"]

    df["mfcc_9_8"] = df["mfcc_9_mean"] - df["mfcc_8_mean"]

    df["mfcc_10_9"] = df["mfcc_10_mean"] - df["mfcc_9_mean"]

    df["mfcc_11_10"] = df["mfcc_11_mean"] - df["mfcc_10_mean"]

    df["mfcc_12_11"] = df["mfcc_12_mean"] - df["mfcc_11_mean"]

    df["mfcc_13_12"] = df["mfcc_13_mean"] - df["mfcc_12_mean"]

    df["mfcc_13_1"] = df["mfcc_13_mean"] - df["mfcc_1_mean"]

    df["chroma_mean_avg"] = (df["chroma_1_mean"] + df["chroma_2_mean"] + df["chroma_3_mean"]+ df["chroma_4_mean"]+ df["chroma_5_mean"]+ df["chroma_6_mean"]+ df["chroma_7_mean"]+ df["chroma_8_mean"]+ df["chroma_9_mean"]+ df["chroma_10_mean"]+ df["chroma_11_mean"]+ df["chroma_12_mean"]) / 12

    df["chroma_mean_var"] = (df["chroma_1_std"] + df["chroma_2_std"] + df["chroma_3_std"]+ df["chroma_4_std"]+ df["chroma_5_std"]+ df["chroma_6_std"]+ df["chroma_7_std"]+ df["chroma_8_std"]+ df["chroma_9_std"]+ df["chroma_10_std"]+ df["chroma_11_std"]+ df["chroma_12_std"]) / 12

    df['spectral_centroid_var'] = df['spectral_centroid_std'] ** 2

    df['spectral_rolloff_var'] = df['spectral_rolloff_std'] ** 2

    df["mfcc_chroma_mean"] = df["mfcc_mean_avg"] * df["chroma_mean_avg"]

    df["mfcc_chroma_std"] = df["mfcc_mean_var"] * df["chroma_mean_var"]

    df["rms_energy_fluctuation"] = df["rms_mean"] * df["rms_std"]

    df["rms_tempo_ratio"] = df["rms_mean"] / df["tempo"]

    df["spectral_diversity_score"] = df["spectral_centroid_std"] + df["spectral_rolloff_std"]

    df["zcr_tempo_ratio"] = df["zcr_mean"] / df["tempo"]

    df["high_freq_intensity"] = (df["spectral_centroid_mean"] + df["spectral_rolloff_mean"]) / 2

    df["spectral_energy_variation"] = df["spectral_centroid_var"] + df["spectral_rolloff_var"]


    cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)

    #cat_cols = [col for col in cat_cols if "stuttering" not in col]

    ####Log Transformation
    log_cols = ["mfcc_1_std", "mfcc_3_std", "mfcc_5_std", "mfcc_7_std", "mfcc_10_std", "rms_mean", "rms_std", "zcr_mean", "zcr_std", "spectral_centroid_mean", "spectral_rolloff_mean", "spectral_centroid_std", "spectral_rolloff_std", "tempo", "zcr_tempo_ratio", "rms_tempo_ratio", "rms_energy_fluctuation" ]

    df[log_cols] = np.log1p(df[log_cols])

    ####Outliers
    for col in num_cols:
        if check_outlier(df, col):
            replace_with_thresholds(df, col)

    ####Scaling

    X_scaled = StandardScaler().fit_transform(df[num_cols])
    df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)



    cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)

    y = df["stuttering"]
    X = df.drop(["stuttering"], axis=1)

    return X, y

def base_models(X, y, Smote=False):
    print("Base Models....")
    classifiers = [
                   ("CART", DecisionTreeClassifier(class_weight= "balanced")),
                   ("RF", RandomForestClassifier(class_weight= "balanced")),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier(class_weight= "balanced",verbose=-1)),
                   ("SVC", SVC(class_weight='balanced')),
                   ('KNN', KNeighborsClassifier()),
                   ('LR', LogisticRegression(class_weight='balanced')),
                   ('GaussianNB',GaussianNB(var_smoothing = 1e-09)),
                   ("ANN", MLPClassifier(max_iter=1000000, early_stopping=True, n_iter_no_change=10)),
                   ('CatBoost', CatBoostClassifier(verbose=False))
    ]

      # StratifiedKFold ile veriyi katmanlara ayırma
    k = 5
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    smote = SMOTE(random_state=42)
    report = []
    model_report = pd.DataFrame()
    total_confusion_matrix = np.zeros((len(np.unique(y)), len(np.unique(y))), dtype=int)

    # Çapraz doğrulama için döngü
    for name, classifier in classifiers:
            print(f"###################################{name}  Smote={Smote}##################################")
            for train_index, test_index in kf.split(X, y):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                if Smote:
                  smote = SMOTE(sampling_strategy=0.63, random_state=42)
                  X_train, y_train = smote.fit_resample(X_train, y_train)

                # Modeli tanımlayıp eğitme
                print("Train.......")
                model = classifier.fit(X_train, y_train)

                # Test verisi ile tahmin yapma
                y_pred = model.predict(X_test)

                # Metrikleri hesaplayıp listeye ekleyin
                accuracies.append(accuracy_score(y_test, y_pred))
                precisions.append(precision_score(y_test, y_pred, average="weighted"))
                recalls.append(recall_score(y_test, y_pred, average="weighted"))
                f1s.append(f1_score(y_test, y_pred, average="weighted"))

                # Sonuçların ortalamalarını yazdırma
                cr = classification_report(y_test, y_pred, output_dict=True)
                print(classification_report(y_test, y_pred))
                report.append(cr)
                cm = confusion_matrix(y_test, y_pred)
                total_confusion_matrix += cm
                # 5. Confusion matrix'i yazdırın
                print("Confusion Matrix:")
                print(cm)



            print(f"Average Accuracy: {np.mean(accuracies)}")
            print(f"Average Precision: {np.mean(precisions)}")
            print(f"Average Recall: {np.mean(recalls)}")
            print(f"Average F1-Score: {np.mean(f1s)}")
            print(total_confusion_matrix)
            model = classifier.fit(X, y)
            if name not in ['KNN', 'GaussianNB', 'LR', "ANN", 'Adaboost', "SVC"]:
                  plot_importance(model, X,num=len(X))

knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200],
                  "colsample_bytree": [0.5, 1]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500],
                   "colsample_bytree": [0.7, 1]}

gnb_params = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
}

adaboost_params = {
    'n_estimators': [50, 100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.5, 1],
    'base_estimator': [DecisionTreeClassifier(max_depth=d) for d in [1, 2, 3, 4]]
}

ann_param = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001],
    'learning_rate': ['constant', 'adaptive']
}

svc_params = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto', 0.1, 1]
}


log_params = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'solver': ['liblinear', 'saga'],
}



classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(class_weight='balanced'), cart_params),
               ('LightGBM', LGBMClassifier(class_weight='balanced', verbose=-1), lightgbm_params),
               ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
               ('GaussianNB',GaussianNB(),gnb_params),
               ("RF", RandomForestClassifier(class_weight='balanced'), rf_params),
               ("SVC", SVC(class_weight='balanced'), svc_params),
               ('LR', LogisticRegression(class_weight='balanced'),log_params),
               ("ANN", MLPClassifier(ann_param)),
               ('Adabost', AdaBoostClassifier(),adaboost_params)
                ]


best_classifiers = [ ('KNN', KNeighborsClassifier(n_neighbors= 45)),
                      ("CART", DecisionTreeClassifier(max_depth= 3, min_samples_split= 2)),
                      ('LightGBM', LGBMClassifier(colsample_bytree=1, learning_rate=0.01, n_estimators=500, verbose=-1, class_weight = 'balanced')),
                      #('XGBoost', XGBClassifier(colsample_bytree=0.5, learning_rate=0.1, max_depth=8, n_estimators=100, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=ratio)),
                      ('GaussianNB',GaussianNB(var_smoothing = 1e-09)),
                      ('Adabost', AdaBoostClassifier(n_estimators=50, random_state=42)),
                      ('Rf', RandomForestClassifier(max_depth=None, max_features=7, min_samples_split=15, n_estimators=300, class_weight = 'balanced')),
                      ("ANN", MLPClassifier(activation= "relu", alpha = 0.001, hidden_layer_sizes = (50,), learning_rate = "adaptive", max_iter=1000, batch_size = 1000))
                    ]

best_classifiers_list =  [KNeighborsClassifier(n_neighbors= 45),
                          DecisionTreeClassifier(max_depth= 3, min_samples_split= 2),
                          LGBMClassifier(colsample_bytree=1, learning_rate=0.01, n_estimators=500, verbose=-1),
                          #XGBClassifier(colsample_bytree=0.5, learning_rate=0.1, max_depth=8, n_estimators=100, use_label_encoder=False, eval_metric='logloss'),
                          GaussianNB(var_smoothing = 1e-09),
                          AdaBoostClassifier(n_estimators=50, random_state=42),
                          RandomForestClassifier(max_depth=None, max_features=7, min_samples_split=15, n_estimators=300),
                          MLPClassifier(activation= "relu", alpha = 0.001, hidden_layer_sizes = (50,), learning_rate = "adaptive", max_iter=1000, batch_size = 1000)
                          ]

def hyperparameter_optimization(X, y, cv=3, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"#######################################################{name}###############################################")
        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)



        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

def best_models(X, y, Smote=False, best_classifiers= best_classifiers):
    print("Best Models....")

    # StratifiedKFold ile veriyi katmanlara ayırma
    k = 5
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    smote = SMOTE(random_state=42)
    report = []
    model_report = pd.DataFrame()
    total_confusion_matrix = np.zeros((len(np.unique(y)), len(np.unique(y))), dtype=int)

    # Çapraz doğrulama için döngü


    for name, classifier in best_classifiers:
            print(f"################################### {name}  Smote={Smote} ##################################")
            for train_index, test_index in kf.split(X, y):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                if Smote:
                  smote = SMOTE(sampling_strategy=0.63, random_state=42)
                  X_train, y_train = smote.fit_resample(X_train, y_train)

                # Modeli tanımlayıp eğitme
                print("Train.......")
                model = classifier.fit(X_train, y_train)

                # Test verisi ile tahmin yapma

                y_pred = model.predict(X_test)

                # Metrikleri hesaplayıp listeye ekleyin
                accuracies.append(accuracy_score(y_test, y_pred))
                precisions.append(precision_score(y_test, y_pred, average="weighted"))
                recalls.append(recall_score(y_test, y_pred, average="weighted"))
                f1s.append(f1_score(y_test, y_pred, average="weighted"))

                # Sonuçların ortalamalarını yazdırma
                cr = classification_report(y_test, y_pred, output_dict=True)
                print(classification_report(y_test, y_pred))
                report.append(cr)
                cm = confusion_matrix(y_test, y_pred)
                total_confusion_matrix += cm
                # 5. Confusion matrix'i yazdırın
                print("Confusion Matrix:")
                print(cm)


            model_report = pd.DataFrame(report)
            print(f"Average Accuracy: {np.mean(accuracies)}")
            print(f"Average Precision: {np.mean(precisions)}")
            print(f"Average Recall: {np.mean(recalls)}")
            print(f"Average F1-Score: {np.mean(f1s)}")
            print(total_confusion_matrix)
            if name not in ['KNN', 'GaussianNB', 'LR', "ANN", 'Adaboost', "SVC"]:
                  plot_importance(model, X,num=len(X))
            #print(model_report)

def voting_classifier(best_classifiers, X, y):
    model_1 = VotingClassifier(estimators = best_classifiers, voting='soft')

    model_2 = VotingClassifier(estimators = best_classifiers, voting='hard')

    models = [model_1]
    smote = SMOTE(random_state=42)
    report = []
    model_report = pd.DataFrame()
    total_confusion_matrix = np.zeros((len(np.unique(y)), len(np.unique(y))), dtype=int)
    #print(classifiers)
    # Çapraz doğrulama için döngü
    for model in models:
        print(f"###################################Voting Classifier...{model}##################################")
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            smote = SMOTE(sampling_strategy=0.63, random_state=42)
            X_smote, y_smote = smote.fit_resample(X_train, y_train)

            # Modeli tanımlayıp eğitme
            print("Voting Classifier...")
            model = model.fit(X_smote, y_smote)


            y_pred = model.predict(X_test)






            accuracies.append(accuracy_score(y_test, y_pred))
            precisions.append(precision_score(y_test, y_pred))
            recalls.append(recall_score(y_test, y_pred))
            f1s.append(f1_score(y_test, y_pred))

            # Sonuçların ortalamalarını yazdırma
            cr = classification_report(y_test, y_pred, output_dict=True)
            print(classification_report(y_test, y_pred))
            report.append(cr)
            cm = confusion_matrix(y_test, y_pred)
            total_confusion_matrix += cm
            # 5. Confusion matrix'i yazdırın
            print("Confusion Matrix:")
            print(cm)

        model_report = pd.DataFrame(report)

        print(f"Average Accuracy: {np.mean(accuracies)}")
        print(f"Average Precision: {np.mean(precisions)}")
        print(f"Average Recall: {np.mean(recalls)}")
        print(f"Average F1-Score: {np.mean(f1s)}")
        print(total_confusion_matrix)
        print(model_report)
        print("################################################################____HARD_____########################################################################")
        smote = SMOTE(sampling_strategy=0.63, random_state=42)
        X_smote, y_smote = smote.fit_resample(X, y)
        voting_clf = model.fit(X_smote, y_smote)
    return voting_clf

def main():
    df = pd.read_excel("/content/drive/MyDrive/Miuul bootcamp/stuttering_dataset.xlsx")
    X, y = stuttering_data_prep(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    base_models(X, y) #Smote = False istersen True yap
    best_models = hyperparameter_optimization(X, y)
    print(pd.DataFrame(best_models))
    voting_clf = voting_classifier(best_models, X, y)
    plot_importance(metot, X,num=len(X))
    joblib.dump(voting_clf, "voting_clf.pkl")



if __name__ == "__main__":
    print("İşlem başladı")
    main()



