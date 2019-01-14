%% Classifiers -> Machine Learning Algorithms

clc; 
clear all;
close all;

%% Load the Feature-Set for both PD & Healthy Patients
% load 'train.mat'
% load 'pd_features555.mat'

%good1 = pca(train_input);
%good2 = pca(train_input_pd);
healthy_features = train_input;
pd_features = train_input_pd555;
training_size =80;
testing_size = 20;
len=680398;

%% Classifiers
[Accuracy_SVM, cMat,p1,f1] = SVM_classifier(healthy_features, pd_features, training_size, testing_size,len);
 [Accuracy_RF,cMat2,p2,f2] = RandomForest_classifier(healthy_features, pd_features, training_size, testing_size,len);
%[Accuracy_NB,cMat3,p3,f3] = NaiveBayes_classifer(healthy_features, pd_features, training_size, testing_size,len);
[Accuracy_KNN,cMat4,p4,f4] = KNN_classifier(healthy_features, pd_features, training_size, testing_size,len);
[Accuracy_AdaB,cMat5,p5,f5] = AdaBoost_classifier(healthy_features, pd_features, training_size, testing_size,len);
[Accuracy_LogitB,cMat6,p6,f6] = LogitBoost_classifier(healthy_features, pd_features, training_size, testing_size,len);
[Accuracy_DTree,cMat7,p7,f7] = DescTree_classifier(healthy_features, pd_features, training_size, testing_size,len);
