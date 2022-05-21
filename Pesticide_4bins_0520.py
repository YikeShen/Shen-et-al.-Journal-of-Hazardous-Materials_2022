import numpy as np
import pickle as pk
import pandas as pd
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.stats import gmean
import scipy
import seaborn as sns
import csv

from rdkit import RDConfig
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
import rdkit
import os
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
from os import listdir

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score

# Categories-Label4: 0-4;4-7; 7-14; >14

#first we generate 1-to-1 matching from cid to molecules & from name to cid

all_files = listdir('SDF')

len(all_files)

#CID matching molecular structures
mol_pool = []
cid2mol = dict()
for f in all_files:
    if f!='.DS_Store':
        temp = f.split('_')[2][:-4]
        mol = Chem.SDMolSupplier('SDF/'+f,removeHs= False)
        cid2mol[str(temp)]=mol[0]
        mol_pool.append(mol[0])

all_names = pd.read_csv('Pesticide_CID_CASID_033022.csv')

all_names

#turn columns to list
names = list(all_names['name'])
cid_name = list(all_names['cid_name'])
cid_name = [str(v) for v in cid_name]
cas_name = list(all_names['cas_name'])

#index searching, name->cid
name2cid = dict()
for i,v in enumerate(names):
    name2cid[v]=cid_name[i]

#index searching, cid->name
cid2name = dict()
for i,v in enumerate(cid_name):
    cid2name[v] = names[i]

#index searching, name->casid
name2cas = dict()
for i,v in enumerate(names):
    name2cas[v]=cas_name[i]

#now we clean up the data, remove cold storage and SMILES not available and variation>5

data = pd.read_csv('pestcide_data.csv')

pestcide_class = data['target class']
compartment_class = data['compartment class']
plant_class= data['plant class']
temperature = data['temperature']

pestcide_class_count = Counter(data['target class'])
compartment_count = Counter(data['compartment'])
compartment_class_count = Counter(data['compartment class'])
plant_class_count = Counter(data['plant class'])
cold_storage_count = Counter(data['cold storage'])

#load data as list
rawdata = []
with open('pestcide_data.csv','r') as csvfile:
    raw = csv.reader(csvfile)
    for line in raw:
        rawdata.append(line)

#remove all cold storage
data_no_cold = []
for i in rawdata[1:]:
    if i[-1] != 'x' and i[-1] != 'x4c'and i[-1] != 'x 4C' and i[-1] != 'x0-2C' and i[-1] != 'x0c':
        data_no_cold.append(i[1:2]+[i[3]]+i[8:10]+[i[5]])

#combine replication experiment with different half-lives
data_combined = []

for i in range(len(data_no_cold)):
    exist = 0
    for j in data_combined:
        if data_no_cold[i][0:4] == j[0:4]:
            j.append(data_no_cold[i][-1])
            exist = 1
    if(not exist):
        data_combined.append(data_no_cold[i])

#calculate geometric mean of replication experiment
half_life_gmean = []
for i in data_combined:
    half_life_gmean.append(gmean([float(n) for n in i[4:]]))

#calculate the variation and test cutoff
variation = []
for i in data_combined:
    variation.append(np.var([float(n) for n in i[4:]]))

#Quality control remove data with variation >5
###### Change test to different variation days ####
pre_screened_data = []
pre_screened_half_life_gmean = []
for i,v in enumerate(data_combined):
    if variation[i]<5:#float('inf')
        pre_screened_data.append(v)
        pre_screened_half_life_gmean.append(half_life_gmean[i])

#tested variations [1,3,4,5,6,7,10]

#remove pesticides that don't have SMILES from pubchem and inorganic 
final_data = []
final_label = []
for j in range(len(pre_screened_data)):
    if pre_screened_data[j][0] in name2cid.keys():
            final_data.append(pre_screened_data[j])
            final_label.append(pre_screened_half_life_gmean[j])
    else:
        pre_screened_data[j][0]
        #print(pre_screened_data[j][0])

#Transform labels to OneHot encoded features
compartment_class_label = {'crop (interior)':[1,0,0,0],'crop surface':[0,1,0,0],'n/a':[0,0,1,0],'root':[0,0,0,1]}
plant_class_label = {'cereals':[1,0,0,0,0,0,0,0,0,0],'forest trees':[0,1,0,0,0,0,0,0,0,0],'fruits':[0,0,1,0,0,0,0,0,0,0],
       'herbs':[0,0,0,1,0,0,0,0,0,0],'leafy crops':[0,0,0,0,1,0,0,0,0,0],'ornamentals': [0,0,0,0,0,1,0,0,0,0],
        'other': [0,0,0,0,0,0,1,0,0,0],'root crops': [0,0,0,0,0,0,0,1,0,0],'vegetables':[0,0,0,0,0,0,0,0,1,0],
        'weeds':[0,0,0,0,0,0,0,0,0,1]}

#Generate ECFP and final input features
features_ecfp4 = []
features_ecfp2 = []
features_ecfp6 = []
features = []
label = []

label12 = []
label4 =[]
selected_cid = set()

for i in range(len(final_data)):
    dd = final_data[i]
    #remove no temperature datapoints
    if dd[1]!='x' and dd[1]!='':  
        
        if final_label[i]<=4:
            label4.append(0)
        elif 4<final_label[i]<=7:
            label4.append(1)
        elif 7<final_label[i]<=14:
            label4.append(2)
        elif 14<final_label[i]:
            label4.append(3)
        
        mol_name = dd[0]
        cid_temp = name2cid[mol_name]
        mol_temp = cid2mol[cid_temp]
        
        plant_class = dd[2]
        temp = [float(dd[1])]
        compartment_class = dd[3]

        plant_label = plant_class_label[plant_class]
        compartment_label = compartment_class_label[compartment_class]
        
        selected_cid.add(cid_temp)
        
        #ECFP features
        ecfp4 = np.array(AllChem.GetMorganFingerprintAsBitVect(mol_temp,2,nBits=1024,useChirality=True)).tolist()
        ecfp2 = np.array(AllChem.GetMorganFingerprintAsBitVect(mol_temp,1,nBits=1024,useChirality=True)).tolist()
        ecfp6 = np.array(AllChem.GetMorganFingerprintAsBitVect(mol_temp,3,nBits=1024,useChirality=True)).tolist()
        
        #Final features combined
        features_ecfp4.append(temp+ecfp4+plant_label+compartment_label)
        features_ecfp2.append(temp+ecfp2+plant_label+compartment_label)
        features_ecfp6.append(temp+ecfp6+plant_label+compartment_label)

features_ecfp4 = np.array(features_ecfp4)
features_ecfp2 = np.array(features_ecfp2)
features_ecfp6 = np.array(features_ecfp6)

feature_ecfp4_z = scipy.stats.mstats.zscore(features_ecfp4,0)
feature_ecfp2_z = scipy.stats.mstats.zscore(features_ecfp2,0)
feature_ecfp6_z = scipy.stats.mstats.zscore(features_ecfp6,0)

#z-transformation NAN to 0
feature_ecfp4_z[np.isnan(feature_ecfp4_z)] = 0
feature_ecfp2_z[np.isnan(feature_ecfp2_z)] = 0
feature_ecfp6_z[np.isnan(feature_ecfp6_z)] = 0

n_sample = len(features_ecfp4)

def Kfold(length,fold):
    size = np.arange(length).tolist()
    train_index = []
    val_index = []
    rest = length % fold
    fold_size = int(length/fold)
    temp_fold_size = fold_size
    for i in range(fold):
        temp_train = []
        temp_val = []
        if rest>0:
            temp_fold_size = fold_size+1
            rest = rest -1
            temp_val = size[i*temp_fold_size:+i*temp_fold_size+temp_fold_size]
            temp_train = size[0:i*temp_fold_size] + size[i*temp_fold_size+temp_fold_size:]
        else:
            temp_val = size[(length % fold)*temp_fold_size+(i-(length % fold))*fold_size
                            :(length % fold)*temp_fold_size+(i-(length % fold))*fold_size+fold_size]
            temp_train = size[0:(length % fold)*temp_fold_size+(i-(length % fold))*fold_size] + size[(length % fold)*temp_fold_size+(i-(length % fold))*fold_size+fold_size:]
        train_index.append(temp_train)
        val_index.append(temp_val)
    return (train_index,val_index)

# Machine Learning Models Run
SVC, NN, GBRT, RF

## Supporting Vector Classifier

svc_all = []
svc_all_bins = []
svc_f1_all = []

for epoch in range(5):
    print('epoch is',epoch)
    #shuffle dataset
    total_id = np.arange(n_sample)
    np.random.shuffle(total_id)
    train_split_index,test_split_index = Kfold(len(features_ecfp4),5)
    splits = 5
    test_score_all_svc = []
    test_score_all_svc_f1 = []
    test_score_all_svc_bin =[]
    
    for k in range(splits):

        print('batch is ',k)
        train_index = train_split_index[k][:int(len(train_split_index[k])*0.875)]
        valid_index = train_split_index[k][int(len(train_split_index[k])*0.875):]
        test_index = test_split_index[k]

        train_id = [total_id[i] for i in train_index]
        valid_id = [total_id[i] for i in valid_index]
        test_id = [total_id[i] for i in test_index]

        train_feature = [feature_ecfp4_z[i] for i in train_id]
        train_label = [label4[i] for i in train_id]

        valid_feature = [feature_ecfp4_z[i] for i in valid_id]
        valid_label = [label4[i] for i in valid_id]

        test_feature = [feature_ecfp4_z[i] for i in test_id]
        test_label = [label4[i] for i in test_id]

        G_pool = [0.00001,0.0001,0.001, 0.01, 0.1, 1,10,100]

        C_pool = [0.0001,0.001, 0.01, 0.1, 1, 10,25,50,100,1000]

        best_valid_score = float('-inf')
        for c in C_pool:
            for g in G_pool:
                model = SVC(kernel='rbf',C=c,gamma=g)#,class_weight='balanced'
                model.fit(train_feature,train_label)
                valid_score = model.score(valid_feature,valid_label)

                if valid_score>best_valid_score:
                    best_valid_score = valid_score
                    test_score = model.score(test_feature,np.array(test_label).reshape(-1,1))
                    pred = model.predict(test_feature)
                    best_n = c
                    best_d = g

        print('test score is',test_score)
        test_score_all_svc.append(test_score)
        test_score_all_svc_f1.append(f1_score(test_label,pred,average='weighted'))
        test_score_all_svc_bin.append(f1_score(test_label,pred,average=None))
        print('f1 is',f1_score(test_label,pred,average='weighted'))
        print('Test score for each category',f1_score(test_label,pred,average=None))
        print('best c is',best_n)
        print('best g is',best_d)
    print('mean accuracy in this epoch',np.mean(test_score_all_svc))
    print('f1 mean accuracy in this epoch',np.mean(test_score_all_svc_f1))
    svc_all.append(np.mean(test_score_all_svc))
    svc_f1_all.append(np.mean(test_score_all_svc_f1))
    svc_all_bins.append(test_score_all_svc_bin)
    

## Random Forest

rf_all = []
rf_all_bins = []
rf_f1_all = []
for epoch in range(5):
    print('epoch is',epoch)
    total_id = np.arange(n_sample)
    np.random.shuffle(total_id)
    train_split_index,test_split_index = Kfold(len(features_ecfp4),5)
    splits = 5
    test_score_all_rf = []
    test_score_all_rf_f1 = []
    test_score_all_rf_bin =[]
    
    for k in range(splits):

        print('batch is ',k)
        train_index = train_split_index[k][:int(len(train_split_index[k])*0.875)]
        valid_index = train_split_index[k][int(len(train_split_index[k])*0.875):]
        test_index = test_split_index[k]

        train_id = [total_id[i] for i in train_index]
        valid_id = [total_id[i] for i in valid_index]
        test_id = [total_id[i] for i in test_index]

        train_feature = [features_ecfp4[i] for i in train_id]
        train_label = [label4[i] for i in train_id]

        valid_feature = [features_ecfp4[i] for i in valid_id]
        valid_label = [label4[i] for i in valid_id]

        test_feature = [features_ecfp4[i] for i in test_id]
        test_label = [label4[i] for i in test_id]

        n_estimator = [50,100,200,300,400,500]
        max_depths = [5,6,7,8,10,15]

        best_valid_score = 0
        for ne in n_estimator:
            for m_d in max_depths:
                model = RandomForestClassifier(n_estimators=ne,max_depth=m_d,class_weight='balanced')
                model.fit(np.array(train_feature),np.array(train_label).reshape(-1))
                valid_score = model.score(valid_feature,np.array(valid_label).reshape(-1,1))
                #print(valid_score)
                if valid_score>best_valid_score:
                    best_valid_score = valid_score
                    test_score = model.score(test_feature,np.array(test_label).reshape(-1,1))
                    pred = model.predict(test_feature)
                    best_n = ne
                    best_d = m_d
        
        print('test score is',test_score)
        test_score_all_rf.append(test_score)
        test_score_all_rf_f1.append(f1_score(test_label,pred,average='weighted'))
        test_score_all_rf_bin.append(f1_score(test_label,pred,average=None))
        print('f1 is',f1_score(test_label,pred,average='weighted'))
        print('Test score for each category',f1_score(test_label,pred,average=None))
        print('best n_estimator is',best_n)
        print('best depth is',best_d)
    print('mean accuracy in this epoch',np.mean(test_score_all_rf))
    print('f1 mean accuracy in this epoch',np.mean(test_score_all_rf_f1))
    rf_all.append(np.mean(test_score_all_rf))
    rf_f1_all.append(np.mean(test_score_all_rf_f1))
    rf_all_bins.append(test_score_all_rf_bin)
                    

## Gradient Boosted Decision Tree (GBDT)

gbdt_all = []
gbdt_all_bins = []
gbdt_f1_all = []
feature_importance_all =[]
for epoch in range(5):
    print('epoch is',epoch)
    #shuffle dataset
    total_id = np.arange(n_sample)
    np.random.shuffle(total_id)
    train_split_index,test_split_index = Kfold(len(features_ecfp4),5)
    splits = 5
    test_score_all_gbdt = []
    test_score_all_gbdt_f1 = []
    test_score_all_gbdt_bin =[]

    pred_all = []
    pred_true = []
    feature_importance = []
    for k in range(splits):

        print('batch is ',k)
        train_index = train_split_index[k][:int(len(train_split_index[k])*0.875)]
        valid_index = train_split_index[k][int(len(train_split_index[k])*0.875):]
        test_index = test_split_index[k]

        train_id = [total_id[i] for i in train_index]
        valid_id = [total_id[i] for i in valid_index]
        test_id = [total_id[i] for i in test_index]

        train_feature = [features_ecfp4[i] for i in train_id]
        train_label = [label4[i] for i in train_id]

        valid_feature = [features_ecfp4[i] for i in valid_id]
        valid_label = [label4[i] for i in valid_id]

        test_feature = [features_ecfp4[i] for i in test_id]
        test_label = [label4[i] for i in test_id]

        n_estimator = [50,100,200,300,500]
        max_depths = [2,3,4,5,6,7]

        best_valid_score = 0
        for ne in n_estimator:
            for m_d in max_depths:
                model = GradientBoostingClassifier(n_estimators=ne,max_depth=m_d,learning_rate=0.05)
                model.fit(np.array(train_feature),np.array(train_label).reshape(-1))
                valid_score = model.score(valid_feature,np.array(valid_label).reshape(-1,1))
                #print(valid_score)
                if valid_score>best_valid_score:
                    best_valid_score = valid_score
                    test_score = model.score(test_feature,np.array(test_label).reshape(-1,1))
                    pred = model.predict(test_feature)
                    #pred = model.predict(test_feature)
                    best_n = ne
                    best_d = m_d
        
        print('test score is',test_score)
        pred_all.append(pred)
        pred_true.append(test_label)
        test_score_all_gbdt.append(test_score)
        test_score_all_gbdt_f1.append(f1_score(test_label,pred,average='weighted'))
        test_score_all_gbdt_bin.append(f1_score(test_label,pred,average=None))
        print('f1 is',f1_score(test_label,pred,average='weighted'))
        feature_importance.append(model.feature_importances_)
        print('best n_estimator is',best_n)
        print('best depth is',best_d)
    print('mean accuracy in this epoch',np.mean(test_score_all_gbdt))
    print('Test score for each category',f1_score(test_label,pred,average=None))
    print('f1 mean accuracy in this epoch',np.mean(test_score_all_gbdt_f1))
    gbdt_all.append(np.mean(test_score_all_gbdt))
    gbdt_f1_all.append(np.mean(test_score_all_gbdt_f1))
    feature_importance_all.append(np.mean(feature_importance,0))
    gbdt_all_bins.append(test_score_all_gbdt_bin)
    

## Logistic Regression

lr_all = []
lr_all_bins = []
lr_f1_all = []
for epoch in range(5):
    print('epoch is',epoch)
    #shuffle dataset
    total_id = np.arange(n_sample)
    np.random.shuffle(total_id)
    train_split_index,test_split_index = Kfold(len(features_ecfp4),5)
    splits = 5
    test_score_all_lr = []
    test_score_all_lr_f1 = []
    test_score_all_lr_bin =[]

    for k in range(splits):

        print('batch is ',k)
        train_index = train_split_index[k][:int(len(train_split_index[k])*0.875)]
        valid_index = train_split_index[k][int(len(train_split_index[k])*0.875):]
        test_index = test_split_index[k]

        train_id = [total_id[i] for i in train_index]
        valid_id = [total_id[i] for i in valid_index
                   ]
        test_id = [total_id[i] for i in test_index]

        train_feature = [feature_ecfp4_z[i] for i in train_id]
        train_label = [label4[i] for i in train_id]

        valid_feature = [feature_ecfp4_z[i] for i in valid_id]
        valid_label = [label4[i] for i in valid_id]

        test_feature = [feature_ecfp4_z[i] for i in test_id]
        test_label = [label4[i] for i in test_id]

        best_valid_score = 0

        model = LogisticRegression(class_weight="balanced",max_iter=1000)
        model.fit(np.array(train_feature),np.array(train_label).reshape(-1))
        valid_score = model.score(valid_feature,np.array(valid_label).reshape(-1,1))
        if valid_score>best_valid_score:
            best_valid_score = valid_score
            pred = model.predict(test_feature)
            test_score = model.score(test_feature,np.array(test_label).reshape(-1,1))



        print('test score is',test_score)
        test_score_all_lr.append(test_score)
        test_score_all_lr_f1.append(f1_score(test_label,pred,average='weighted'))
        test_score_all_lr_bin.append(f1_score(test_label,pred,average=None))
        print('f1 is',f1_score(test_label,pred,average='weighted'))

    print('mean accuracy in this epoch',np.mean(test_score_all_lr))
    print('Test score for each category',f1_score(test_label,pred,average=None))
    print('f1 mean accuracy in this epoch',np.mean(test_score_all_lr_f1))
    lr_all.append(np.mean(test_score_all_lr))
    lr_f1_all.append(np.mean(test_score_all_lr_f1))
    lr_all_bins.append(test_score_all_lr_bin)
    

gbdt_all_ = str(np.mean(gbdt_all).round(3))+u"\u00B1"+str(np.std(gbdt_all).round(3))
gbdt_f1_all_ = str(np.mean(gbdt_f1_all).round(3))+u"\u00B1"+str(np.std(gbdt_f1_all).round(3))

rf_all_ = str(np.mean(rf_all).round(3))+u"\u00B1"+str(np.std(rf_all).round(3))
rf_f1_all_ = str(np.mean(rf_f1_all).round(3))+u"\u00B1"+str(np.std(rf_f1_all).round(3))

svc_all_ = str(np.mean(svc_all).round(3))+u"\u00B1"+str(np.std(svc_all).round(3))
svc_f1_all_ = str(np.mean(svc_f1_all).round(3))+u"\u00B1"+str(np.std(svc_f1_all).round(3))

lr_all_ = str(np.mean(lr_all).round(3))+u"\u00B1"+str(np.std(lr_all).round(3))
lr_f1_all_ = str(np.mean(lr_f1_all).round(3))+u"\u00B1"+str(np.std(lr_f1_all).round(3))



d1 = {'GBRT-ECFP4':gbdt_all_,'LR-ECFP4':lr_all_,'SVC-ECFP4':svc_all_,'RF-ECFP4':rf_all_}
d1

d2 = {'GBRT-ECFP4':gbdt_f1_all_, 'LR-ECFP4':lr_f1_all_,'SVC-ECFP4':svc_f1_all_,'RF-ECFP4':rf_f1_all_}
d2

pesticidemodels_df= pd.DataFrame([d1])
pesticidemodels_df.loc[len(pesticidemodels_df.index)] = d2
pesticidemodels_df.index=['F1 macro','F1 weighted']
pesticidemodels_df=pesticidemodels_df.T
pesticidemodels_df

d3 = {'GBRT-ECFP4_bin3_mean':np.mean(gbdt_all_eachbin,0),'GBRT-ECFP4_bin3_std':np.std(gbdt_all_eachbin,0),      'RF-ECFP4_bin3_mean':np.mean(rf_all_eachbin,0),'RF-ECFP4_bin3_std':np.std(rf_all_eachbin,0),      'SVC-ECFP4_bin3_mean':np.mean(rf_all_eachbin,0),'SVC-ECFP4_bin3_std':np.std(rf_all_eachbin,0),      'LR-ECFP4_bin3_mean':np.mean(rf_all_eachbin,0),'LR-ECFP4_bin3_std':np.std(rf_all_eachbin,0)}
d3
pesticidemodels_df_BIN4= pd.DataFrame(d3)
pesticidemodels_df_BIN4.index=['0-4','4-7','7-14','>14']
pesticidemodels_df_BIN4
#pesticidemodels_df.to_csv('Results/4bins.csv',sep='\t')

