clc;
clear;
close all;

%% ===================== LOAD DATA =====================
% Read the Excel file (Sheet1) and preserve the original column names
T = readtable('Concrete_Data.xls', ...
    'Sheet', 'Sheet1', ...
    'VariableNamingRule', 'preserve');

%% ===================== EXTRACT FEATURES AND TARGET =====================
% Features (inputs) = first 8 columns
X = T{:, 1:8};   % size: 1031 x 8

% ---- Manual feature scaling: divide columns 1,2,3,4,6,7 by 100 ----
colsToScale = [1,2,3,4,6,7];
X(:,colsToScale) = X(:,colsToScale) / 100;

% Transpose to match your GMDH code (features x samples)
X = X';

% Target (output) = last column (9th)
Y = T{:, 9}';

%% ===================== TRAIN / VALIDATION SPLIT =====================
trainRatio = 0.7;
numSamples = size(X,2);
numTrain = round(trainRatio * numSamples);

Xtrain = X(:,1:numTrain);
Ytrain = Y(1:numTrain);

Xval = X(:,numTrain+1:end);
Yval = Y(numTrain+1:end);

%% ===================== USER INPUT =====================
Lmax = input('Enter maximum number of layers: ');
MaxNeurons = input('Enter max neurons per layer: ');

%% ===================== INITIALIZATION =====================
Xlayer = Xtrain;
Xval_layer = Xval;

% ---- normalize training data ----
mu = mean(Xlayer,2);
sigma = std(Xlayer,0,2);
sigma(sigma==0) = 1;

Xlayer = (Xlayer - mu) ./ sigma;

% ---- normalize validation using training statistics ----
Xval_layer = (Xval_layer - mu) ./ sigma;

bestMSE = inf;
patience = 2;
badLayers = 0;

valMSE = zeros(1,Lmax);
Yval_pred_layers = cell(Lmax,1);

%% ===================== GMDH LAYERS =====================
for layer = 1:Lmax

    numFeatures = size(Xlayer,1);
    numTrainSamples = size(Xlayer,2);
    numValSamples = size(Xval_layer,2);
    numNeurons = nchoosek(numFeatures,2);

    if numNeurons == 0
        break;
    end

    Yp_train = zeros(numTrainSamples,numNeurons);
    Yp_val   = zeros(numValSamples,numNeurons);

    r = 1;
    for i = 1:numFeatures
        for j = i+1:numFeatures

            % -------- TRAIN --------
            Xn = [ones(numTrainSamples,1), ...
                  Xlayer(i,:).', ...
                  Xlayer(j,:).', ...
                  (Xlayer(i,:).*Xlayer(j,:)).', ...
                  (Xlayer(i,:).^2).', ...
                  (Xlayer(j,:).^2).'];

            a = pinv(Xn) * Ytrain.';
            Yp_train(:,r) = Xn * a;

            % -------- VALIDATION --------
            Xv = [ones(numValSamples,1), ...
                  Xval_layer(i,:).', ...
                  Xval_layer(j,:).', ...
                  (Xval_layer(i,:).*Xval_layer(j,:)).', ...
                  (Xval_layer(i,:).^2).', ...
                  (Xval_layer(j,:).^2).'];

            Yp_val(:,r) = Xv * a;

            r = r + 1;
        end
    end

    %% ===================== NEURON SELECTION =====================
    valRMSE = sqrt(mean((Yp_val - Yval.').^2,1));
    [sortedRMSE, idx] = sort(valRMSE);

    numKeep = min(MaxNeurons, length(idx));
    keepIdx = idx(1:numKeep);

    %% ===================== UPDATE LAYERS =====================
    Xlayer = Yp_train(:,keepIdx).';
    Xval_layer = Yp_val(:,keepIdx).';

    % ---- normalize training layer ----
    mu = mean(Xlayer,2);
    sigma = std(Xlayer,0,2);
    sigma(sigma==0) = 1;
    Xlayer = (Xlayer - mu) ./ sigma;

    % ---- normalize validation using training mu/sigma ----
    Xval_layer = (Xval_layer - mu) ./ sigma;

    Yval_pred_layers{layer} = Yp_val(:,keepIdx);
    valMSE(layer) = mean(sortedRMSE(1:numKeep).^2);

    fprintf('Layer %d | Validation MSE = %.6f\n', layer, valMSE(layer));

    %% ===================== EARLY STOPPING =====================
    if valMSE(layer) < bestMSE
        bestMSE = valMSE(layer);
        badLayers = 0;
    else
        badLayers = badLayers + 1;
        if badLayers >= patience
            fprintf('Early stopping at layer %d\n', layer);
            break;
        end
    end
end

%% ===================== PLOT =====================
[~,idxSort] = sort(Yval);
Ytrue_sorted = Yval(idxSort);

figure; hold on;
colors = jet(layer);

for l = 1:layer
    Ypred = mean(Yval_pred_layers{l},2);
    plot(Ypred(idxSort),'Color',colors(l,:),'LineWidth',1.5);
end

plot(Ytrue_sorted,'k--','LineWidth',2);
grid on;
xlabel('Validation Sample (sorted)');
ylabel('Output (MPa)');
title('GMDH Validation Prediction per Layer');
legend([arrayfun(@(x) sprintf('Layer %d',x),1:layer,'UniformOutput',false),{'True'}]);
%% ===================== PLOT BEST LAYER =====================
% Find the best layer (lowest validation MSE)
[~, bestLayer] = min(valMSE(1:layer));

Ybest_pred = mean(Yval_pred_layers{bestLayer}, 2);

figure; hold on;
plot(Ybest_pred(idxSort), 'b-', 'LineWidth', 2);
plot(Ytrue_sorted, 'k--', 'LineWidth', 2);
grid on;
xlabel('Validation Sample (sorted)');
ylabel('Output (MPa)');
title(sprintf('GMDH Best Layer Prediction (Layer %d)', bestLayer));
legend('Best Layer Prediction', 'True Output');