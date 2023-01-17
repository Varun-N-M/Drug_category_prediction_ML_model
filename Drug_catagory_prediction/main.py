import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder,StandardScaler

from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,learning_curve

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.decomposition import PCA

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.cluster import KMeans

from sklearn.metrics import accuracy_score,mean_squared_error

import xgboost

df = pd.read_csv('drug200.csv')

print(df.head().to_string())
print(df.info())
print(df.describe().to_string())


def grab_col_names(dataframe, car_th=10, cat_th=20):
    cat_col = [col for col in dataframe.columns if dataframe[col].dtype == 'O']
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtype != 'O']
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > cat_th and dataframe[col].dtype == 'O']

    cat_col = cat_col + num_but_cat
    cat_col = [col for col in cat_col if col not in cat_but_car]

    num_col = [col for col in dataframe.columns if dataframe[col].dtype != 'O']
    num_col = [col for col in num_col if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_col)}')
    print(f'num_cols: {len(num_col)}')

    return cat_col, num_col, cat_but_car

cat_col, num_col, cat_but_car = grab_col_names(df)

for i in df[cat_col]:
    print(df[i].value_counts())

le = LabelEncoder()

for i in df[cat_col]:
    df[i] = le.fit_transform(df[i])

df = pd.get_dummies(df,columns=['Sex'])
df = pd.get_dummies(df,columns=['BP'])
df = pd.get_dummies(df,columns=['Cholesterol'])

print(df.head().to_string())

b_plt = df.boxplot(vert =False)
plt.show()


corr = df.corr()
f,ax = plt.subplots(figsize = (10,10))
c_plot = sns.heatmap(corr,annot = True)
plt.show()

def VIF(independent_variables):
    vif = pd.DataFrame()
    vif['vif'] = [variance_inflation_factor(independent_variables.values,i) for i in range (independent_variables.shape[1])]
    vif['independent_variables']= independent_variables.columns
    vif = vif.sort_values(by=['vif'],ascending=False)
    return vif

print(VIF(df.drop('Drug',axis=1)))
#
def CWT (data, tcol):
    independent_variables = data.drop(tcol, axis=1).columns
    corr_result = []
    for col in independent_variables :
        corr_result.append(data[tcol].corr(data[col]))
    result = pd.DataFrame([independent_variables, corr_result], index=['independent variables', 'correlation']).T    #T is for transpose
    return result.sort_values(by = 'correlation',ascending = False)

print(CWT(df,'Drug'))
#
def PCA_1(x):
    n_comp = len(x.columns)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # Applying PCA
    for i in range(1, n_comp):
        pca = PCA(n_components=i)
        p_comp = pca.fit_transform(x)
        evr = np.cumsum(pca.explained_variance_ratio_)
        if evr[i - 1] > 0.9:
            n_components = i
            break
    print('Explained varience ratio after pca is: ', evr)
    # creating a pcs dataframe
    col = []
    for j in range(1, n_components + 1):
        col.append('PC_' + str(j))
    pca_df = pd.DataFrame(p_comp, columns=col)
    return pca_df

transformed_df = PCA_1(df.drop('Drug',axis=1))

transformed_df = transformed_df.join(df['Drug'],how = 'left')
print(transformed_df.head().to_string())
#
def train_and_test_split(data,t_col, testsize=0.3):
    x = data.drop(t_col, axis=1)
    y = data[t_col]
    return train_test_split(x,y,test_size=testsize, random_state=1)

def model_builder(model_name, estimator, data, t_col):
    x_train,x_test,y_train,y_test = train_and_test_split(data, t_col)
    estimator.fit(x_train, y_train)
    y_pred = estimator.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    return [model_name, accuracy, rmse]

def multiple_models(data,t_col):
    col_names = ['model_name', 'accuracy_score','RMSE']
    result = pd.DataFrame(columns=col_names)
    result.loc[len(result)] = model_builder('LogisticRegression',LogisticRegression(),data,t_col)
    result.loc[len(result)] = model_builder('DecisionTreeClassifier',DecisionTreeClassifier(),data,t_col)
    result.loc[len(result)] = model_builder('KneighborClassifier',KNeighborsClassifier(),data,t_col)
    result.loc[len(result)] = model_builder('RandomForestClassifier',RandomForestClassifier(),data,t_col)
    result.loc[len(result)] = model_builder('SVC',SVC(),data,t_col)
    result.loc[len(result)] = model_builder('AdaBoostClassifier',AdaBoostClassifier(),data,t_col)
    result.loc[len(result)] = model_builder('GradientBoostingClassifier',GradientBoostingClassifier(),data,t_col)
    result.loc[len(result)] = model_builder('XGBClassifier',XGBClassifier(),data,t_col)
    return result.sort_values(by='accuracy_score',ascending=False)
print( )
print('Set 1')
print(multiple_models(transformed_df,'Drug'))

def kfoldCV(x, y, fold=10):
    score_lr = cross_val_score(LogisticRegression(), x, y, cv=fold)
    score_dt = cross_val_score(DecisionTreeClassifier(), x, y, cv=fold)
    score_kn = cross_val_score(KNeighborsClassifier(), x, y, cv=fold)
    score_rf = cross_val_score(RandomForestClassifier(), x, y, cv=fold)
    score_svc = cross_val_score(SVC(), x, y, cv=fold)
    score_ab = cross_val_score(AdaBoostClassifier(), x, y, cv=fold)
    score_gb = cross_val_score(GradientBoostingClassifier(), x, y, cv=fold)
    score_xb = cross_val_score(XGBClassifier(), x, y, cv=fold)

    model_names = ['Logisticregression', 'DecisionTreeClassifier', 'KNeighborsClassifier', 'RandomForestClassifier',
                   'SVC', 'AdaBoostClassifier', 'GradientBoostingClassifier', 'XGBClassifier']
    scores = [score_lr, score_dt, score_kn, score_rf, score_svc, score_ab, score_gb, score_xb]
    result = []
    for i in range(len(model_names)):
        score_mean = np.mean(scores[i])
        score_std = np.std(scores[i])
        m_names = model_names[i]
        temp = [m_names, score_mean, score_std]
        result.append(temp)
    kfold_df = pd.DataFrame(result, columns=['model_names', 'cv_score', 'cv_std'])
    return kfold_df.sort_values(by='cv_score', ascending=False)
print( )
print(kfoldCV(transformed_df.drop('Drug',axis = 1),transformed_df['Drug']))
print( )
def tuning(x,y,fold = 10):
   #parameters grids for different models
    param_dtc = {'criterion':['gini', 'entropy', 'log_loss'],'max_depth':[3,5,7,9,11],'max_features':[1,2,3,4,5,6,7,'auto','log2', 'sqrt']}
    param_knn = {'weights':['uniform','distance'],'algorithm':['auto','ball_tree','kd_tree','brute']}
    param_svc = {'gamma':['scale','auto'],'C': [0.1,1,1.5,2]}
    param_rf = {'max_depth':[3,5,7,9,11],'max_features':[1,2,3,4,5,6,7,'auto','log2', 'sqrt'],'n_estimators':[50,100,150,200]}
    param_ab = {'n_estimators':[50,100,150,200],'learning_rate':[0.1,0.5,0.7,1,5,10,20,50,100]}
    param_gb = {'n_estimators':[50,100,150,200],'learning_rate':[0.1,0.5,0.7,1,5,10,20,50,100]}
    param_xb = {'eta':[0.1,0.5,10.7,1,5,10,20],'max_depth':[3,5,7,9,10],'gamma':[0,10,20,50],'reg_lambda':[0,1,3,5,7,10],'alpha':[0,1,3,5,7,10]}
    #Creating Model object
    tune_dtc = GridSearchCV(DecisionTreeClassifier(),param_dtc,cv=fold)
    tune_knn = GridSearchCV(KNeighborsClassifier(),param_knn,cv=fold)
    tune_svc = GridSearchCV(SVC(),param_svc,cv=fold)
    tune_rf = GridSearchCV(RandomForestClassifier(),param_rf,cv=fold)
    tune_ab = GridSearchCV(AdaBoostClassifier(),param_ab,cv=fold)
    tune_gb = GridSearchCV(GradientBoostingClassifier(),param_gb,cv=fold)
    tune_xb = GridSearchCV(XGBClassifier(),param_xb,cv=fold)
    #Model fitting
    tune_dtc.fit(x,y)
    tune_knn.fit(x,y)
    tune_svc.fit(x,y)
    tune_rf.fit(x,y)
    tune_ab.fit(x,y)
    tune_gb.fit(x,y)
    tune_xb.fit(x,y)

    tune = [tune_rf,tune_xb,tune_gb,tune_knn,tune_svc,tune_dtc,tune_ab]
    models = ['RF','XB','GB','KNN','SVR','DTR','AB']
    for i in range(len(tune)):
        print('model:',models[i])
        print('Best_params:',tune[i].best_params_)

print(tuning(transformed_df.drop('Drug',axis=1),transformed_df['Drug']))

def cv_post_hpt(x,y,fold = 10):
    score_lr = cross_val_score(LogisticRegression(),x,y,cv= fold)
    score_dt = cross_val_score(DecisionTreeClassifier(criterion= 'gini', max_depth= 11, max_features= 4),x,y,cv= fold)
    score_kn = cross_val_score(KNeighborsClassifier(weights ='distance', algorithm ='auto'),x,y,cv=fold)
    score_rf = cross_val_score(RandomForestClassifier(max_depth= 9, max_features=7, n_estimators= 150),x,y,cv=fold)
    score_svc = cross_val_score(SVC(gamma='auto', C= 1.5),x,y,cv=fold)
    score_ab = cross_val_score(AdaBoostClassifier(n_estimators= 200, learning_rate=0.5),x,y,cv=fold)
    score_gb = cross_val_score(GradientBoostingClassifier(n_estimators=500, learning_rate= 0.1),x,y,cv=fold)
    score_xb = cross_val_score(XGBClassifier(eta=1,max_depth= 9,gamma=0,reg_lambda = 5,alpha=0),x,y,cv=fold)

    model_names = ['LogisticRegression','RandomForestClassifier','DecisionTreeClassifier','KNeighborsClassifier','SVC','AdaBoostClassifier','GradientBoostingClassifier','XGBClassifier']
    scores = [score_lr,score_rf, score_dt,score_kn,score_svc,score_ab,score_gb,score_xb]
    result = []
    for i in range(len(model_names)):
        score_mean = np.mean(scores[i])
        score_std = np.std(scores[i])
        m_names = model_names[i]
        temp = [m_names,score_mean,score_std]
        result.append(temp)
    kfold_df = pd.DataFrame(result,columns=['model_names','cv_score','cv_std'])
    return kfold_df.sort_values(by='cv_score',ascending=False)

print(cv_post_hpt(transformed_df.drop('Drug',axis=1),transformed_df['Drug']))
print( )
labels = KMeans(n_clusters = 2,random_state=2)
clusters = labels.fit_predict(df.drop('Drug',axis=1))

def clustering(x,tcol,clusters):
    column = list(set(list(x.columns)) - set(list('Drug')))
    #column = list(x.column)
    r = int(len(column)/2)
    if len(column)%2 == 0:
        r=r
    else:
        r += 1      #same as r+1
    f,ax = plt.subplots(r,2,figsize = (15,15))
    a = 0
    for row in range(r):
        for col in range(2):
            if a!= len(column):
                ax[row][col].scatter(x[tcol] , x[column[a]], c = clusters)
                ax[row][col].set_xlabel(tcol)
                ax[row][col].set_ylabel(column[a])
                a += 1
x = df.drop('Drug',axis = 1)
for col in x.columns:
    clustering(x , col , clusters)
plt.show()


new_df = df.join(pd.DataFrame(clusters,columns=['cluster']),how = 'left')
new_f = new_df.groupby('cluster')['Age'].agg(['mean','median'])

cluster_df = new_df.merge(new_f, on = 'cluster',how= 'left')
print(cluster_df.head().to_string())

print('Set-2')
print(multiple_models(cluster_df,'Drug'))
print( )
print(kfoldCV(transformed_df.drop('Drug',axis = 1),transformed_df['Drug']))
print( )
print(cv_post_hpt(cluster_df.drop('Drug',axis=1),cluster_df['Drug']))
print( )

new__df = cluster_df

rfe = RFE(estimator = RandomForestClassifier())

rfe.fit(new__df.drop('Drug',axis=1),new__df['Drug'])

print(rfe.support_)

print(new__df.columns)

final_df = cluster_df[['Age', 'Na_to_K','BP_0', 'BP_1', 'BP_2','Cholesterol_1','Drug']]
print(final_df)
#
print('Set-3')
print( )
print(multiple_models(final_df,'Drug'))
print( )
print(kfoldCV(final_df.drop('Drug',axis = 1),final_df['Drug']))
print( )
print(cv_post_hpt(final_df.drop('Drug',axis=1),final_df['Drug']))
print( )
x_train,x_test,y_train,y_test = train_and_test_split(cluster_df,'Drug')

xgb = XGBClassifier()
xgb.fit(x_train,y_train)         #parameters and hyperparameters
#
xgboost.plot_importance(xgb)
plt.show()
#
subset_df = cluster_df[['Na_to_K','Age','Cholesterol_0','BP_0','BP_1','BP_2','Sex_0','Drug']]
print('Set-4')
print( )
print(multiple_models(subset_df,'Drug'))
print( )
print(kfoldCV(subset_df.drop('Drug',axis = 1),subset_df['Drug']))
print( )
print(cv_post_hpt(subset_df.drop('Drug',axis=1),subset_df['Drug']))
print( )
def generate_learning_curve(model_name,estimater,x,y):
    train_size,train_score,test_score = learning_curve(estimater,x,y,cv= 10)
#     print('train_size',train_size)
#     print('train_score',train_score)
#     print('test_score',test_score)
    train_score_mean = np.mean(train_score,axis=1)
    test_score_mean = np.mean(test_score,axis=1)
    plt.plot(train_size,train_score_mean, c = 'blue')
    plt.plot(train_size,test_score_mean, c = 'red')
    plt.xlabel('Samples')
    plt.ylabel('Scores')
    plt.title('Learning curve for '+model_name)
    plt.legend(('Training accuray','Testing accuracy'))


model_names = [LogisticRegression(), RandomForestClassifier(), DecisionTreeClassifier(), KNeighborsClassifier(), SVC(),
               AdaBoostClassifier(), GradientBoostingClassifier(), XGBClassifier()]
for i, model in enumerate(model_names):
#     print(i)
#     print(model_names[i])
    fig = plt.figure(figsize=(6,3))
    ax = fig.add_subplot(5,2,i+1)
    generate_learning_curve(type(model).__name__,model,cluster_df.drop('Drug',axis=1),cluster_df['Drug'])
plt.show()