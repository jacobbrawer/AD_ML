#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 16:30:41 2019

Alzheimer’s Disease Classification using ML Pipeline on Fast Fourier Transformed EEG Data

@authors: Jacob Brawer and Tyler Yoshihara
"""

import warnings
warnings.filterwarnings("ignore")
import os
import numpy, matplotlib, scipy, sklearn, statsmodels, parfit, scikitplot, pandas, nilearn
from sklearn import preprocessing, model_selection, linear_model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from imblearn.over_sampling import SMOTE, ADASYN
from nilearn import plotting as niplot
from nilearn.masking import compute_epi_mask
from nilearn.input_data import NiftiMasker
from nilearn.decoding import SpaceNetClassifier
from nipype.interfaces import spm
from nipype.interfaces import matlab
import matplotlib.pyplot as plt
import pandas
import numpy as np

HCvMCI = pandas.read_csv('C:/Users/jacob/Documents/Spring 2019 Semester/Machine Learning with Neural Signal/Final Project/Alzheimers Data for Project/MCIvsHCFourier.csv')
MCIvAD = pandas.read_csv('C:/Users/jacob/Documents/Spring 2019 Semester/Machine Learning with Neural Signal/Final Project/Alzheimers Data for Project/MCIvsADFourier.csv')
ADvHC = pandas.read_csv('C:/Users/jacob/Documents/Spring 2019 Semester/Machine Learning with Neural Signal/Final Project/Alzheimers Data for Project/ADvsHCFourier.csv')

HCvMCI = np.asarray(HCvMCI)
MCIvAD = np.asarray(MCIvAD)
ADvHC = np.asarray(ADvHC)

HCvMCI = HCvMCI[:,1:]
MCIvAD = MCIvAD[:,1:]
ADvHC = ADvHC[:,1:]

YHvM = HCvMCI[:,304]
YMvA = MCIvAD[:,304]
YAvH = ADvHC[:,304]

HCvMCI = HCvMCI[:,0:304]
MCIvAD = MCIvAD[:,0:304]
ADvHC = ADvHC[:,0:304]

HCvMCI = preprocessing.scale(HCvMCI, axis=0)
MCIvAD = preprocessing.scale(MCIvAD, axis=0)
ADvHC = preprocessing.scale(ADvHC, axis=0)

freq_bands = scipy.signal.welch(ADvHC, fs=256)


# Set up Train, Val, Test sets for all datasets

[train_inds, test_inds] = next(
        model_selection.ShuffleSplit(
                test_size=0.2,random_state=100).split(
                        HCvMCI,y=YHvM
                        )
        )

XTempTrain = HCvMCI[train_inds,]

# Split Train Set into Train and Validation Sets
[train2_inds, val_inds] = next(
        model_selection.ShuffleSplit(
                test_size=0.4,random_state=100).split(
                        XTempTrain,y=YHvM[train_inds]
                        )
        )

# Form the indices to select for each set
TrainInds = train_inds[train2_inds]
ValInds = train_inds[val_inds]
TestInds = test_inds


# Create sets of X and Y data using indices  for HCvMCI
    
XTrainHvM = HCvMCI[TrainInds,]
YTrainHvM = YHvM[TrainInds]
XValHvM = HCvMCI[ValInds,]
YValHvM = YHvM[ValInds]
XTestHvM = HCvMCI[TestInds,]
YTestHvM = YHvM[TestInds]


# Set up Train, Val, Test sets for MCI vs AD
[train_inds, test_inds] = next(
        model_selection.ShuffleSplit(
                test_size=0.2,random_state=100).split(
                        MCIvAD,y=YMvA
                        )
        )

XTempTrain = MCIvAD[train_inds,]

# Split Train Set into Train and Validation Sets
[train2_inds, val_inds] = next(
        model_selection.ShuffleSplit(
                test_size=0.4,random_state=100).split(
                        XTempTrain,y=YMvA[train_inds]
                        )
        )

# Form the indices to select for each set
TrainInds = train_inds[train2_inds]
ValInds = train_inds[val_inds]
TestInds = test_inds


# Create sets of X and Y data using indices  for HCvMCI
    
XTrainMvA = MCIvAD[TrainInds,]
YTrainMvA = YMvA[TrainInds]
XValMvA = MCIvAD[ValInds,]
YValMvA = YMvA[ValInds]
XTestMvA = MCIvAD[TestInds,]
YTestMvA = YMvA[TestInds]



# Set up Train, Val, Test sets for AD vs HC
[train_inds, test_inds] = next(
        model_selection.ShuffleSplit(
                test_size=0.2,random_state=100).split(
                        ADvHC,y=YAvH
                        )
        )

XTempTrain = MCIvAD[train_inds,]

# Split Train Set into Train and Validation Sets

[train2_inds, val_inds] = next(
        model_selection.ShuffleSplit(
                test_size=0.4,random_state=100).split(
                        XTempTrain,y=YAvH[train_inds]
                        )
        )

# Form the indices to select for each set
TrainInds = train_inds[train2_inds]
ValInds = train_inds[val_inds]
TestInds = test_inds

# Create sets of X and Y data using indices  for HCvMCI
    
XTrainAvH = ADvHC[TrainInds,]
YTrainAvH = YAvH[TrainInds]
XValAvH = ADvHC[ValInds,]
YValAvH = YAvH[ValInds]
XTestAvH = ADvHC[TestInds,]
YTestAvH = YAvH[TestInds]



#%%
#Running RVC - HCvMCI

#Reintegrate the validation set into the train/test sets for RVC

RXTrainHvM = HCvMCI[train_inds,]
RYTrainHvM = YHvM[train_inds]
RXTestHvM = HCvMCI[test_inds,]
RYTestHvM = YHvM[test_inds]

#resampling w/ SMOTE to account for uneven sampling

[XTrainResHvM,YTrainResHvM] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(RXTrainHvM,RYTrainHvM)
[XTestResHvM,YTestResHvM] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(RXTestHvM,RYTestHvM)

from skrvm import RVC
RVCMod = RVC(kernel = 'linear',
             verbose = True)
RVCMod.fit(XTrainResHvm,YTrainResHvM)

#create feature importance evaluation function

def RVMFeatImp(RVs):
    NumRVs = RVs.shape[0]
    SumD = 0
    for RVNum in range(1,NumRVs):
        d1 = RVs[RVNum-1,]
        d2 = sum(numpy.ndarray.flatten(
                RVs[numpy.int8(
                        numpy.setdiff1d(numpy.linspace(0,NumRVs-1,NumRVs),RVNum))]))
        SumD = SumD + (d1/d2)
    SumD = SumD/NumRVs
    return SumD


RVs = RVCMod.relevance_
DVals = RVMFeatImp(RVs)

RVCPred1 = RVCMod.predict_proba(XTestResHvM)
RVCPred2 = RVCMod.predict(XTestResHvM)

# Plot Receiver Operating Characteristic (ROC) Curve
scikitplot.metrics.plot_roc(YTestResHvM,RVCPred1, title = 'HCvMCI: RVC')
# Plot the Confusion Matrix for additional insight
scikitplot.metrics.plot_confusion_matrix(YTestResHvM,RVCPred2)

#%%
# Running RLR - HCvMCI

#Testing for multicollinearity 

coef1 = np.corrcoef(HCvMCI, rowvar = False)
plt.hist(coef1)

#resampling w/ SMOTE to account for uneven sampling

[XTrainRLRHvM,YTrainRLRHvM] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XTrainHvM,YTrainHvM)
[XValRLRHvM,YValRLRHvM] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XValHvM,YValHvM)
[XTestRLRHvM,YTestRLRHvM] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XTestHvM,YTestHvM)

#run gridsearch for hyperparameters

ncores = 2
grid = {
    'C': numpy.linspace(1e-10,1e5,num = 100), #Inverse lambda
    'penalty': ['l2']
}
paramGrid = sklearn.model_selection.ParameterGrid(grid)
RLRMod = sklearn.linear_model.LogisticRegression(tol = 1e-10,
                                                random_state = 100,
                                                n_jobs = ncores,
                                                verbose = 1)
[bestModel,bestScore,allModels,allScores] = parfit.bestFit(RLRMod,
paramGrid = paramGrid,               
X_train = XTrainRLRHvM,
y_train = YTrainRLRHvM,
X_val = XValRLRHvM,
y_val = YValRLRHvM,
metric = sklearn.metrics.roc_auc_score,
n_jobs = ncores,
scoreLabel = 'AUC')

# Test on Test Set
RLRTestPred = bestModel.predict_proba(XTestRLRHvM)
RLRTestPred2 = bestModel.predict(XTestRLRHvM)

# Plot Receiver Operating Characteristic (ROC) Curve
scikitplot.metrics.plot_roc(YTestHvM,RLRTestPred,title = 'LR with LASSO')

# Plot the Confusion Matrix for additional insight
scikitplot.metrics.plot_confusion_matrix(YTestRLRHvM,RLRTestPred2)


# %%
# RF - HCvMCI

# Define the Hyperparameter grid values
# Number of trees in random forest
n_estimators = [int(x) for x in numpy.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in numpy.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Code for the Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier 

# Declare the Classifier
RFC = RandomForestClassifier(criterion = 'gini') #can use 'entropy' instead

#Resampling w/ SMOTE to account for uneven sampling group

[XTrainRFHvM,YTrainRFHvM] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XTrainHvM,YTrainHvM)
[XValRFHvM,YValRFHvM] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XValHvM,YValHvM)
[XTestRFHvM,YTestRFHvM] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XTestHvM,YTestHvM)

# Raw Classifier

from sklearn.model_selection import RandomizedSearchCV
# n_iter is the number of randomized parameter combinations tried
RFC_RandomSearch = RandomizedSearchCV(estimator = RFC,
                                      param_distributions = random_grid,
                                      n_iter = 100, cv = 3, verbose=2,
                                      random_state=10, n_jobs = 2)
RFC_RandomSearch.fit(XTrainRFHvM,YTrainRFHvM)

# Look at the Tuned "Best" Parameters
RFC_RandomSearch.best_params_
RFC_RandomSearch.best_score_
RFC_RandomSearch.best_estimator_.feature_importances_

# Fit using the best parameters
# Look at the Feature Importance
FeatImp = RFC_RandomSearch.best_estimator_.feature_importances_
NZInds = numpy.nonzero(FeatImp)
num_NZInds = len(NZInds[0])
Keep_NZVals = [x for x in FeatImp[NZInds[0]] if 
               (abs(x) >= numpy.mean(FeatImp[NZInds[0]]) 
               + 4*numpy.std(FeatImp[NZInds[0]]))]
ThreshVal = numpy.mean(FeatImp[NZInds[0]]) + 2*numpy.std(FeatImp[NZInds[0]])
Keep_NZInds = numpy.nonzero(abs(FeatImp[NZInds[0]]) >= ThreshVal)
Final_NZInds = NZInds[0][Keep_NZInds]



Pred1_S2 = RFC_RandomSearch.best_estimator_.predict(XTestRFHvM)
Pred2_S2 = RFC_RandomSearch.best_estimator_.predict_proba(XTestRFHvM)

scikitplot.metrics.plot_roc(YTestRFHvM,Pred2_S2, title = 'HCvMC RF')
scikitplot.metrics.plot_confusion_matrix(YTestRFHvM,Pred1_S2)

#%%
#Running RVC - MCIvAD


#Reintegrate the validation set into the train/test sets for RVC

RXTrainMvA = MCIvAD[train_inds,]
RYTrainMvA = YMvA[train_inds]
RXTestMvA = MCIvAD[test_inds,]
RYTestMvA = YMvA[test_inds]

#resampling w/ SMOTE to account for uneven sampling

[XTrainResMvA,YTrainResMvA] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(RXTrainMvA,RYTrainMvA)
[XTestResMvA,YTestResMvA] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(RXTestMvA,RYTestMvA)

from skrvm import RVC
RVCMod = RVC(kernel = 'linear',
             verbose = True)
RVCMod.fit(RXTrainMvA,RYTrainMvA)

#create feature importance evaluation function

def RVMFeatImp(RVs):
    NumRVs = RVs.shape[0]
    SumD = 0
    for RVNum in range(1,NumRVs):
        d1 = RVs[RVNum-1,]
        d2 = sum(numpy.ndarray.flatten(
                RVs[numpy.int8(
                        numpy.setdiff1d(numpy.linspace(0,NumRVs-1,NumRVs),RVNum))]))
        SumD = SumD + (d1/d2)
    SumD = SumD/NumRVs
    return SumD


RVs = RVCMod.relevance_
DVals = RVMFeatImp(RVs)

RVCPred1 = RVCMod.predict_proba(XTestResMvA)
RVCPred2 = RVCMod.predict(XTestResMvA)

# Plot Receiver Operating Characteristic (ROC) Curve
scikitplot.metrics.plot_roc(YTestResMvA,RVCPred1, title = 'MCIvAD: RVC')
# Plot the Confusion Matrix for additional insight
scikitplot.metrics.plot_confusion_matrix(YTestResMvA,RVCPred2)

#%%
# Running RLR - HCvMCI

#Testing for multicollinearity 

coef2 = np.corrcoef(MCIvAD, rowvar = False)
plt.hist(coef2)

#resampling w/ SMOTE to account for uneven sampling

[XTrainRLRMvA,YTrainRLRMvA] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XTrainMvA,YTrainMvA)
[XValRLRMvA,YValRLRMvA] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XValMvA,YValMvA)
[XTestRLRMvA,YTestRLRMvA] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XTestMvA,YTestMvA)

#gridsearch for hyperparameter evaluation

ncores = 2
grid = {
    'C': numpy.linspace(1e-10,1e5,num = 100), #Inverse lambda
    'penalty': ['l2']
}
paramGrid = sklearn.model_selection.ParameterGrid(grid)
RLRMod = sklearn.linear_model.LogisticRegression(tol = 1e-10,
                                                random_state = 100,
                                                n_jobs = ncores,
                                                verbose = 1)
[bestModel,bestScore,allModels,allScores] = parfit.bestFit(RLRMod,
paramGrid = paramGrid,               
X_train = XTrainRLRMvA,
y_train = YTrainRLRMvA,
X_val = XValRLRMvA,
y_val = YValRLRMvA,
metric = sklearn.metrics.roc_auc_score,
n_jobs = ncores,
scoreLabel = 'AUC')

# Test on Test Set
RLRTestPred = bestModel.predict_proba(XTestRLRMvA)
RLRTestPred2 = bestModel.predict(XTestRLRMvA)

# Plot Receiver Operating Characteristic (ROC) Curve
scikitplot.metrics.plot_roc(YTestRLRMvA,RLRTestPred,title = 'LR with LASSO')

# Plot the Confusion Matrix for additional insight
scikitplot.metrics.plot_confusion_matrix(YTestRLRMvA,RLRTestPred2)


# %%
# RF - HCvMCI

# Define the Hyperparameter grid values
# Number of trees in random forest
n_estimators = [int(x) for x in numpy.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in numpy.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Code for the Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier 

# Declare the Classifier
RFC = RandomForestClassifier(criterion = 'gini') #can use 'entropy' instead


#Resampling w/ SMOTE to account for uneven sampling group

[XTrainRFMvA,YTrainRFMvA] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XTrainMvA,YTrainMvA)
[XValRFAvA,YValRFMvA] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XValMvA,YValMvA)
[XTestRFMvA,YTestRFMvA] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XTestMvA,YTestMvA)


# Raw Classifier

from sklearn.model_selection import RandomizedSearchCV
# n_iter is the number of randomized parameter combinations tried
RFC_RandomSearch = RandomizedSearchCV(estimator = RFC,
                                      param_distributions = random_grid,
                                      n_iter = 100, cv = 3, verbose=2,
                                      random_state=10, n_jobs = 2)
RFC_RandomSearch.fit(XTrainRFMvA,YTrainRFMvA)

# Look at the Tuned "Best" Parameters
RFC_RandomSearch.best_params_
RFC_RandomSearch.best_score_
RFC_RandomSearch.best_estimator_.feature_importances_

# Fit using the best parameters
# Look at the Feature Importance
FeatImp = RFC_RandomSearch.best_estimator_.feature_importances_
NZInds = numpy.nonzero(FeatImp)
num_NZInds = len(NZInds[0])
Keep_NZVals = [x for x in FeatImp[NZInds[0]] if 
               (abs(x) >= numpy.mean(FeatImp[NZInds[0]]) 
               + 4*numpy.std(FeatImp[NZInds[0]]))]
ThreshVal = numpy.mean(FeatImp[NZInds[0]]) + 2*numpy.std(FeatImp[NZInds[0]])
Keep_NZInds = numpy.nonzero(abs(FeatImp[NZInds[0]]) >= ThreshVal)
Final_NZInds = NZInds[0][Keep_NZInds]



Pred1_S2 = RFC_RandomSearch.best_estimator_.predict(XTestRFMvA)
Pred2_S2 = RFC_RandomSearch.best_estimator_.predict_proba(XTestRFMvA)

scikitplot.metrics.plot_roc(YTestRFMvA,Pred2_S2, title = 'MCIvAD RF')
scikitplot.metrics.plot_confusion_matrix(YTestRFMvA,Pred1_S2)

#%%
#Running RVC - ADvHC

#Reintegrate the validation set into the train/test sets for RVC

RXTrainAvH = ADvHC[train_inds,]
RYTrainAvH = YAvH[train_inds]
RXTestAvH = ADvHC[test_inds,]
RYTestAvH = YAvH[test_inds]

#Resampling w/ SMOTE to account for uneven sampling groups

[XTrainResAvH,YTrainResAvH] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(RXTrainAvH,RYTrainAvH)
[XTestResAvH,YTestResAvH] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(RXTestAvH,RYTestAvH)


from skrvm import RVC
RVCMod = RVC(kernel = 'linear',
             verbose = True)
RVCMod.fit(XTrainResAvH,YTrainResAvH)

#create feature importance evaluation function

def RVMFeatImp(RVs):
    NumRVs = RVs.shape[0]
    SumD = 0
    for RVNum in range(1,NumRVs):
        d1 = RVs[RVNum-1,]
        d2 = sum(numpy.ndarray.flatten(
                RVs[numpy.int8(
                        numpy.setdiff1d(numpy.linspace(0,NumRVs-1,NumRVs),RVNum))]))
        SumD = SumD + (d1/d2)
    SumD = SumD/NumRVs
    return SumD


RVs = RVCMod.relevance_
FeatImp_RVC = RVMFeatImp(RVs)

RVCPred1 = RVCMod.predict_proba(XTestResAvH)
RVCPred2 = RVCMod.predict(XTestResAvH)
# Evaluate Performance (DON'T RELY ON ACCURACY!!!)
# Plot Receiver Operating Characteristic (ROC) Curve
scikitplot.metrics.plot_roc(YTestResAvH,RVCPred1, title = 'ADvHC: RVC')
# Plot the Confusion Matrix for additional insight
scikitplot.metrics.plot_confusion_matrix(YTestResAvH,RVCPred2)

#%%
# Running RLR - HCvMCI

#Testing for multicollinearity 

coef3 = np.corrcoef(ADvHC, rowvar = False)
plt.hist(coef3)

#resampling w/ SMOTE to account for uneven sampling

[XTrainRLRAvH,YTrainRLRAvH] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XTrainAvH,YTrainAvH)
[XValRLRAvH,YValRLRAvH] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XValAvH,YValAvH)
[XTestRLRAvH,YTestRLRAvH] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XTestAvH,YTestAvH)

#gridsearch for hyperparameter evaluation

ncores = 2
grid = {
    'C': numpy.linspace(1e-10,1e5,num = 100), #Inverse lambda
    'penalty': ['l2']
}
paramGrid = sklearn.model_selection.ParameterGrid(grid)
RLRMod = sklearn.linear_model.LogisticRegression(tol = 1e-10,
                                                random_state = 100,
                                                n_jobs = ncores,
                                                verbose = 1)
[bestModel,bestScore,allModels,allScores] = parfit.bestFit(RLRMod,
paramGrid = paramGrid,               
X_train = XTrainRLRAvH,
y_train = YTrainRLRAvH,
X_val = XValRLRAvH,
y_val = YValRLRAvH,
metric = sklearn.metrics.roc_auc_score,
n_jobs = ncores,
scoreLabel = 'AUC')

# Test on Test Set
RLRTestPred = bestModel.predict_proba(XTestRLRAvH)
RLRTestPred2 = bestModel.predict(XTestRLRAvH)

FeatImp_RLR = abs(bestModel.coef_.T)


# Plot Receiver Operating Characteristic (ROC) Curve
scikitplot.metrics.plot_roc(YTestRLRAvH,RLRTestPred,title = 'AD vs HC with Ridge')

# Plot the Confusion Matrix for additional insight
scikitplot.metrics.plot_confusion_matrix(YTestRLRAvH,RLRTestPred2)


# %%
# RF - AD vs HC

# Define the Hyperparameter grid values
# Number of trees in random forest
n_estimators = [int(x) for x in numpy.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in numpy.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Code for the Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier 

# Declare the Classifier
RFC = RandomForestClassifier(criterion = 'gini') #can use 'entropy' instead

#Resampling w/ SMOTE to account for uneven sampling group

[XTrainRFAvH,YTrainRFAvH] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XTrainAvH,YTrainAvH)
[XValRFAvH,YValRFAvH] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XValAvH,YValAvH)
[XTestRFAvH,YTestRFAvH] = SMOTE(random_state = 100,k_neighbors = 3).fit_resample(XTestAvH,YTestAvH)

# Raw Classifier

from sklearn.model_selection import RandomizedSearchCV
# n_iter is the number of randomized parameter combinations tried
RFC_RandomSearch = RandomizedSearchCV(estimator = RFC,
                                      param_distributions = random_grid,
                                      n_iter = 100, cv = 3, verbose=2,
                                      random_state=10, n_jobs = 2)
RFC_RandomSearch.fit(XTrainRFAvH,YTrainRFAvH)

# Look at the Tuned "Best" Parameters
RFC_RandomSearch.best_params_
RFC_RandomSearch.best_score_
RFC_RandomSearch.best_estimator_.feature_importances_

# Fit using the best parameters
# Look at the Feature Importance
FeatImp_RF = RFC_RandomSearch.best_estimator_.feature_importances_
NZInds = numpy.nonzero(FeatImp_RF)
num_NZInds = len(NZInds[0])
Keep_NZVals = [x for x in FeatImp_RF[NZInds[0]] if 
               (abs(x) >= numpy.mean(FeatImp_RF[NZInds[0]]) 
               + 4*numpy.std(FeatImp_RF[NZInds[0]]))]
ThreshVal = numpy.mean(FeatImp_RF[NZInds[0]]) + 2*numpy.std(FeatImp_RF[NZInds[0]])
Keep_NZInds = numpy.nonzero(abs(FeatImp_RF[NZInds[0]]) >= ThreshVal)
Final_NZInds = NZInds[0][Keep_NZInds]



Pred1_S2 = RFC_RandomSearch.best_estimator_.predict(XTestRFAvH)
Pred2_S2 = RFC_RandomSearch.best_estimator_.predict_proba(XTestRFAvH)

scikitplot.metrics.plot_roc(YTestRFAvH,Pred2_S2, title = 'ADvHC RF')
scikitplot.metrics.plot_confusion_matrix(YTestRFAvH,Pred1_S2)


FeatImp_RVC = abs(FeatImp_RVC)

FeatMatrix = np.stack([FeatImp_RVC, FeatImp_RLR, FeatImp_RF], axis = 1)
FeatCorr = np.corrcoef(FeatMatrix.T)
np.triu(FeatCorr)

plt.plot(FeatImp_RVC, FeatImp_RF, 'o')
np.savetxt("FeatImp_RVC_ADvHC.csv", FeatImp_RVC)
np.savetxt("FeatImp_RF_ADvHC.csv", FeatImp_RF)


from scipy import stats

FeatImp_RVC_reshape = numpy.reshape(FeatImp_RVC,[19,16])
FeatImp_RVC_mean = numpy.mean(FeatImp_RVC_reshape, axis=0)
FeatImp_RVC_std = numpy.std(FeatImp_RVC_reshape, axis=0)
conf_int = stats.norm.interval(0.95, loc=FeatImp_RVC_mean, scale=FeatImp_RVC_std)
Freq_values = np.linspace(0,30,16)
plt.plot(Freq_values, FeatImp_RVC_mean, 'o')



FeatImp_RF_reshape = numpy.reshape(FeatImp_RF,[19,16])
FeatImp_RF_mean = numpy.mean(FeatImp_RF_reshape, axis=0)
FeatImp_RF_std = numpy.std(FeatImp_RF_reshape, axis=0)
conf_int = stats.norm.interval(0.95, loc=FeatImp_RF_mean, scale=FeatImp_RF_std)

Freq_values = np.linspace(0,30,16)
plt.plot(Freq_values,FeatImp_RF_mean, 'o')

