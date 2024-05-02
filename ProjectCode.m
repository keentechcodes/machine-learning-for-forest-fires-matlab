% Import the data
data = readtable('/MATLAB Drive/LBOEC3B/Project/Project1/Algerian_forest_fires_dataset_UPDATE.csv');

% Add a 'Row' column to the dataset
rowNames = arrayfun(@(x) sprintf('Row%d', x), 1:height(data), 'UniformOutput', false);
data.Properties.RowNames = rowNames;

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

% Row indices for Bejaia and Sidi-Bel Abbes Regions in the dataset
bejaia_rows = arrayfun(@(x) sprintf('Row%d', x), 3:124, 'UniformOutput', false);
sidi_rows = arrayfun(@(x) sprintf('Row%d', x), 128:249, 'UniformOutput', false);

% Separate the datasets by region using row indices
testingData_bejaia = testingData(ismember(testingData.Row, bejaia_rows), :);
testingData_sidi = testingData(ismember(testingData.Row, sidi_rows), :);

% Define YTest_bejaia and YTest_sidi
YTest_bejaia = testingData_bejaia.Classes;
YTest_sidi = testingData_sidi.Classes;

% Define YPred_knn_bejaia and YPred_knn_sidi
YPred_knn_bejaia = YPred_knn(ismember(testingData.Row, bejaia_rows));
YPred_knn_sidi = YPred_knn(ismember(testingData.Row, sidi_rows));

% Define YPred_linear_bejaia and YPred_linear_sidi
YPred_linear_bejaia = YPred_linear(ismember(testingData.Row, bejaia_rows));
YPred_linear_sidi = YPred_linear(ismember(testingData.Row, sidi_rows));

% Calculate accuracy for each region
accuracy_knn_bejaia = sum(YPred_knn_bejaia == YTest_bejaia) / length(YTest_bejaia);
accuracy_knn_sidi = sum(YPred_knn_sidi == YTest_sidi) / length(YTest_sidi);
accuracy_linear_bejaia = sum(YPred_linear_bejaia == YTest_bejaia) / length(YTest_bejaia);
accuracy_linear_sidi = sum(YPred_linear_sidi == YTest_sidi) / length(YTest_sidi);

% Plotting
figure('Position', [100, 100, 1000, 800]);

subplot(4, 2, [1, 2]);
gscatter(testingData_bejaia.(Variable1), testingData_bejaia.(Variable2), YTest_bejaia);
xlabel(Variable1);
ylabel(Variable2);
title('Bejaia True Classes (k-NN)');
legend('Location', 'best');
axis tight;

subplot(4, 2, [3, 4]);
gscatter(testingData_bejaia.(Variable1), testingData_bejaia.(Variable2), YPred_knn_bejaia);
xlabel(Variable1);
ylabel(Variable2);
title('Bejaia k-NN Predicted Classes');
legend('Location', 'best');
axis tight;

subplot(4, 2, [5, 6]);
gscatter(testingData_sidi.(Variable1), testingData_sidi.(Variable2), YTest_sidi);
xlabel(Variable1);
ylabel(Variable2);
title('Sidi-Bel Abbes True Classes (k-NN)');
legend('Location', 'best');
axis tight;

subplot(4, 2, [7, 8]);
gscatter(testingData_sidi.(Variable1), testingData_sidi.(Variable2), YPred_knn_sidi);
xlabel(Variable1);
ylabel(Variable2);
title('Sidi-Bel Abbes k-NN Predicted Classes');
legend('Location', 'best');
axis tight;

% Display k-NN accuracy
annotation('textbox', [0.05, 0.02, 0.9, 0.04], 'String', ['k-NN Accuracy: ' num2str(accuracy_knn)], 'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold', 'LineStyle', 'none');

% Second figure for linear regression plots
figure('Position', [100, 100, 1000, 800]);

subplot(4, 2, [1, 2]);
gscatter(testingData_bejaia.(Variable1), testingData_bejaia.(Variable2), YTest_bejaia);
xlabel(Variable1);
ylabel(Variable2);
title('Bejaia True Classes (Linear Regression)');
legend('Location', 'best');
axis tight;

subplot(4, 2, [3, 4]);
gscatter(testingData_bejaia.(Variable1), testingData_bejaia.(Variable2), YPred_linear_bejaia);
xlabel(Variable1);
ylabel(Variable2);
title('Bejaia Linear Regression Predicted Classes');
legend('Location', 'best');
axis tight;

subplot(4, 2, [5, 6]);
gscatter(testingData_sidi.(Variable1), testingData_sidi.(Variable2), YTest_sidi);
xlabel(Variable1);
ylabel(Variable2);
title('Sidi-Bel Abbes True Classes (Linear Regression)');
legend('Location', 'best');
axis tight;

subplot(4, 2, [7, 8]);
gscatter(testingData_sidi.(Variable1), testingData_sidi.(Variable2), YPred_linear_sidi);
xlabel(Variable1);
ylabel(Variable2);
title('Sidi-Bel Abbes Linear Regression Predicted Classes');
legend('Location', 'best');
axis tight;

% Display accuracy
annotation('textbox', [0.05, 0.02, 0.9, 0.04], 'String', ['Linear Regression Accuracy: ' num2str(accuracy_linear)], 'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold', 'LineStyle', 'none');

