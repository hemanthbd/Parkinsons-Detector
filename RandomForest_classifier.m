function [accuracy_RF,cMat, precision, f1score] = RandomForest_classifier(healthy_features, pd_features, training_size, testing_size,length)

% Training Phase
train_labels = ones(training_size,1);
train_labels(1:40) = 0;

X_train = zeros(training_size,length);

X_train(1:40,:) = healthy_features(1:40,:);
X_train(41:80,:) = pd_features(1:40,:);

%makes the RF Tree model based on the labels given

rf_fit = TreeBagger(100,X_train,train_labels,'OOBPrediction','On',...
    'Method','classification');

%Testing Phase
test_labels = ones(testing_size,1);
test_labels(1:10) = 0;

X_test = zeros(testing_size,length);

X_test(1:10,:) = healthy_features(41:50,:);
X_test(11:20,:) = pd_features(41:50,:);

% Prediction                            
rf_predict = predict(rf_fit,X_test); % predicts output of identified model Random Forest.

% Confusion Matrix
%Creates the confusion matrix for SVM based on the labels given
cMat = confusionmat(test_labels,str2double(rf_predict)); % returns the confusion matrix of known (labels) and predicted (C1) groups.

accuracy_RF = 100*(cMat(1,1)+cMat(2,2))/testing_size;
precision = 100*(cMat(1,1)/(cMat(1,1)+ cMat(2,1)));
recall = 100*(cMat(1,1)/(cMat(1,1)+ cMat(1,2)));
f1score = 2*(precision*recall)/(precision + recall);
end