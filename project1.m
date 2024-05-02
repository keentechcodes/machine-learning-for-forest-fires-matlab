% Import the data
data = readtable('Algerian_forest_fires_dataset_UPDATE.csv');

% Convert the "Classes" column into categorical data
data.Classes = categorical(data.Classes);

% Remove rows with missing values in the "Classes" column
data = data(~ismissing(data.Classes), :);

% Create a non-stratified holdout partition
% Try with loop for best partitions
rng('default'); % For reproducibility
holdoutRatio = 0.3; % Hold out 30% of the data for testing
cv = cvpartition(height(data), 'HoldOut', holdoutRatio);

% Create the training and testing sets
trainingData = data(cv.training,:);
testingData = data(cv.test,:);

% Separate the predictor variables (features) from the response variable (Classes)
predictorVars = data.Properties.VariableNames(1:end-1); % Assuming "Classes" is the last column in the dataset
XTrain = trainingData{:, predictorVars};
YTrain = trainingData.Classes;
XTest = testingData{:, predictorVars};
YTest = testingData.Classes;

% Normalize the predictor variables
XTrain_normalized = rescale(XTrain, 'InputMin', min(XTrain, [], 1), 'InputMax', max(XTrain, [], 1));
XTest_normalized = rescale(XTest, 'InputMin', min(XTrain, [], 1), 'InputMax', max(XTrain, [], 1));

% Train the k-NN classifier using the normalized data
k = 3; % Choose the number of neighbors
knnModel = fitcknn(XTrain_normalized, YTrain, 'NumNeighbors', k);

% Predict the classes for the testing set using the normalized data
YPred_knn = predict(knnModel, XTest_normalized);

% Evaluate the performance of the k-NN classifier
accuracy_knn = sum(YPred_knn == YTest) / length(YTest);
confusionMatrix_knn = confusionmat(YTest, YPred_knn);

% Convert categorical response variable to numeric
YTrain_numeric = double(YTrain);
YTest_numeric = double(YTest);

% Train the linear regression model using the normalized data
linearModel = fitlm(XTrain_normalized, YTrain_numeric);

% Predict the numeric response for the testing set using the normalized data
YPred_linear_numeric = predict(linearModel, XTest_normalized);

% Convert the numeric predictions back to the original categories
YPred_linear = categorical(round(YPred_linear_numeric), 1:max(YTrain_numeric), categories(YTrain));

% Evaluate the performance of the linear regression model
accuracy_linear = sum(YPred_linear == YTest) / length(YTest);
confusionMatrix_linear = confusionmat(YTest, YPred_linear);

% Choose the first two predictor variables for plotting
Variable1 = predictorVars{1};
Variable2 = predictorVars{2};

% Plot the true and predicted classes for both k-NN and linear regression
figure;
subplot(1, 2, 1);
gscatter(testingData.(Variable1), testingData.(Variable2), YTest);
xlabel(Variable1);
ylabel(Variable2);
title('True Classes');
legend('Location', 'best');

subplot(1, 2, 2);
gscatter(testingData.(Variable1), testingData.(Variable2), YPred_knn);
xlabel(Variable1);
ylabel(Variable2);
title('k-NN Predicted Classes');
legend('Location', 'best');

figure;
subplot(1, 2, 1);
gscatter(testingData.(Variable1), testingData.(Variable2), YTest);
xlabel(Variable1);
ylabel(Variable2);
title('True Classes');
legend('Location', 'best');

subplot(1, 2, 2);
gscatter(testingData.(Variable1), testingData.(Variable2), YPred_linear);
xlabel(Variable1);
ylabel(Variable2);
title('Linear Regression Predicted Classes');
legend('Location', 'best');


