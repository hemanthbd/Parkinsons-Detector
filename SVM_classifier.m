function [accuracy_SVM, cMat, precision, f1score] = SVM_classifier(healthy_features, pd_features, training_size, testing_size,length)

% Training Phase
train_labels = ones(training_size,1);
train_labels(1:40) = 0;

X_train = zeros(training_size,length);

X_train(1:40,:) = healthy_features(1:40,:);

X_train(41:80,:) = pd_features(1:40,:);

%makes the SVM model based on the labels given
svm_fit = fitcsvm(X_train,train_labels);

%Testing Phase
test_labels = ones(testing_size,1);
test_labels(1:10) = 0;

X_test = zeros(testing_size,length);

X_test(1:10,:) = healthy_features(41:50,:);
X_test(11:20,:) = pd_features(41:50,:);

% Prediction                            
svm_predict = predict(svm_fit,X_test); % predicts output of identified model Support Vector Machines.

% Confusion Matrix
%Creates the confusion matrix for SVM based on the labels given
cMat = confusionmat(test_labels',svm_predict'); % returns the confusion matrix of known (labels) and predicted (C1) groups.
accuracy_SVM = 100*(cMat(1,1)+cMat(2,2))/testing_size;
precision = 100*(cMat(1,1)/(cMat(1,1)+ cMat(2,1)));
recall = 100*(cMat(1,1)/(cMat(1,1)+ cMat(1,2)));
f1score = 2*(precision*recall)/(precision + recall);
end