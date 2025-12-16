% final_MLR.m
% End-to-end MLR training + LOSO + synthesized-weight prediction trials
% Save as final_MLR.m and run from MATLAB.
% ---------------------------------------------------------
% Multiple Linear Regression (MLR) – LOSO Evaluation
%
% Task: Regression of load/weight from EEG features
% Method: Ridge-regularized MLR with lagged features
% Validation: K-fold CV + Leave-One-Subject-Out (LOSO)
%
% Input : Feature CSV (precomputed)
% Output: Trained models, RMSE metrics, predictions
% ---------------------------------------------------------


clear; clc; close all;
rng(1);

%% -------- CONFIG (edit as needed) --------
csvfile = 'finalcsv_withkurtosis_normalized_updated.csv'; % input CSV with precomputed features
saveFile = 'mlr_epoch_models_enhanced.mat';               % output MAT
subjectCol = 'subject';
epochCol = 'epoch';      % optional (set '' if absent)
markerCol = 'marker';    % optional (set '' if absent)
targetCol = 'weight';
G = 12;                  % number of lags (0..G)
lags = 0:G;
Kfold = 5;
lambdaGrid = logspace(-8, 0, 40);
standardizePerSubject = true;
doLOSO = true;

% random trials config
numTrials = 10;
randSeedTrials = 2;
randRangeMax = 7.5; % random target weight upper bound

%% -------- READ CSV and basic checks --------
if ~isfile(csvfile)
    error('CSV not found: %s', csvfile);
end
T = readtable(csvfile);
T.Properties.VariableNames = matlab.lang.makeValidName(T.Properties.VariableNames);

if ~any(strcmp(T.Properties.VariableNames, targetCol))
    error('No target column ''%s'' in CSV.', targetCol);
end
if ~any(strcmp(T.Properties.VariableNames, subjectCol))
    error('No subject column ''%s'' in CSV.', subjectCol);
end

hasMarker = ~isempty(markerCol) && any(strcmp(T.Properties.VariableNames, markerCol));
hasEpoch  = ~isempty(epochCol)  && any(strcmp(T.Properties.VariableNames, epochCol));

% select numeric base features (exclude control cols)
exclude = {targetCol, subjectCol};
if hasMarker, exclude{end+1} = markerCol; end
if hasEpoch,  exclude{end+1} = epochCol;  end

isNum = varfun(@isnumeric, T, 'OutputFormat','uniform');
baseCols = T.Properties.VariableNames(isNum);
baseCols = setdiff(baseCols, exclude, 'stable');
if isempty(baseCols)
    error('No numeric base features found to lag. Adjust CSV or column selection.');
end
fprintf('Using %d base features.\n', numel(baseCols));

%% -------- per-subject standardization (optional) --------
T2 = T; % copy
subjects = unique(T2.(subjectCol), 'stable');
if standardizePerSubject
    fprintf('Applying per-subject z-score to base features...\n');
    for i = 1:numel(subjects)
        s = subjects(i);
        idx = ismember(T2.(subjectCol), s);
        if sum(idx) < 2, continue; end
        arr = table2array(T2(idx, baseCols));
        M = mean(arr, 1);
        S = std(arr, 0, 1); S(S==0) = 1;
        T2{idx, baseCols} = (arr - M) ./ S;
    end
end

%% -------- build lagged design matrix per subject --------
X_all = [];
y_all = [];
marker_all = [];
featNames = {};
for si = 1:numel(subjects)
    s = subjects(si);
    idx = find(ismember(T2.(subjectCol), s));
    if hasEpoch
        [~, ord] = sort(T2.(epochCol)(idx));
        idx = idx(ord);
    end
    subTbl = T2(idx, :);
    nSub = height(subTbl);
    nFeat = numel(baseCols);
    nLags = numel(lags);
    subX = nan(nSub, nFeat * nLags);
    for k = 1:nLags
        lag = lags(k);
        if lag == 0
            subX(:, (k-1)*nFeat + (1:nFeat)) = table2array(subTbl(:, baseCols));
        else
            rowsFrom = 1:(nSub - lag);
            rowsTo = (lag+1):nSub;
            if ~isempty(rowsFrom)
                subX(rowsTo, (k-1)*nFeat + (1:nFeat)) = table2array(subTbl(rowsFrom, baseCols));
            end
        end
    end
    valid = all(~isnan(subX), 2) & ~isnan(subTbl.(targetCol));
    X_all = [X_all; subX(valid, :)]; %#ok<AGROW>
    y_all = [y_all; subTbl.(targetCol)(valid)]; %#ok<AGROW>
    if hasMarker
        marker_all = [marker_all; subTbl.(markerCol)(valid)]; %#ok<AGROW>
    else
        marker_all = [marker_all; repmat("GLOBAL", sum(valid), 1)]; %#ok<AGROW>
    end
    if si == 1
        for k = 1:nLags
            for f = 1:nFeat
                featNames{end+1} = sprintf('%s_lag%d', baseCols{f}, lags(k)); %#ok<AGROW>
            end
        end
    end
end
fprintf('Built lagged matrix: samples=%d, features=%d\n', size(X_all,1), size(X_all,2));

%% -------- global standardize (store muX, sX) --------
muX = mean(X_all, 1);
sX = std(X_all, [], 1); sX(sX==0) = 1;
X = (X_all - muX) ./ sX;
y = y_all;

%% -------- train per-segment ridge MLR with CV lambda selection --------
if hasMarker
    segs = unique(marker_all, 'stable');
else
    segs = "GLOBAL";
end
models = struct();
results = struct();

for si = 1:numel(segs)
    seg = segs(si);
    sel = strcmp(string(marker_all), string(seg));
    Xseg = X(sel, :);
    yseg = y(sel);
    fprintf('Training segment %s: samples=%d\n', char(seg), size(Xseg,1));
    nz = std(Xseg,0,1) > 0;
    Xseg = Xseg(:, nz);
    featNamesSeg = featNames(nz);
    mseGrid = nan(numel(lambdaGrid), 1);
    for li = 1:numel(lambdaGrid)
        lam = lambdaGrid(li);
        try
            cvM = fitrlinear(Xseg, yseg, 'Learner','leastsquares', 'Regularization','ridge',...
                'Lambda', lam, 'KFold', Kfold, 'FitBias', true);
            mseGrid(li) = kfoldLoss(cvM, 'LossFun', 'mse');
        catch
            mseGrid(li) = NaN;
        end
    end
    [~,bestI] = min(mseGrid);
    bestLam = lambdaGrid(bestI);
    fprintf(' best lambda = %.4g\n', bestLam);
    mdl = fitrlinear(Xseg, yseg, 'Learner','leastsquares', 'Regularization','ridge', 'Lambda', bestLam, 'FitBias', true);
    try
        cvMbest = fitrlinear(Xseg, yseg, 'Learner','leastsquares','Regularization','ridge',...
            'Lambda', bestLam, 'KFold', Kfold, 'FitBias', true);
        rmse_cv = sqrt(kfoldLoss(cvMbest, 'LossFun', 'mse'));
    catch
        rmse_cv = NaN;
    end
    models.(char(seg)) = mdl;
    results.(char(seg)).bestLambda = bestLam;
    results.(char(seg)).cvMSEgrid = mseGrid;
    results.(char(seg)).rmse_cv = rmse_cv;
    results.(char(seg)).featureNames = featNamesSeg;
    fprintf(' RMSE (CV) = %.4f kg\n\n', rmse_cv);
end

%% -------- LOSO evaluation --------
if doLOSO
    fprintf('Performing LOSO across %d subjects...\n', numel(subjects));
    loso_preds = [];
    loso_truth = [];
    for si = 1:numel(subjects)
        s = subjects(si);
        % build train set excluding subject s
        trainMask = ~ismember(T2.(subjectCol), s);
        Xtrain = [];
        ytrain = [];
        for sj = unique(T2.(subjectCol)(trainMask))'
            idx2 = find(ismember(T2.(subjectCol), sj));
            if hasEpoch, [~,ord2] = sort(T2.(epochCol)(idx2)); idx2 = idx2(ord2); end
            subTbl2 = T2(idx2, :);
            nsub2 = height(subTbl2);
            sXsub = nan(nsub2, numel(baseCols) * numel(lags));
            for k = 1:numel(lags)
                lag = lags(k);
                if lag == 0
                    sXsub(:, (k-1)*numel(baseCols) + (1:numel(baseCols))) = table2array(subTbl2(:, baseCols));
                else
                    rowsFrom = 1:(nsub2 - lag);
                    rowsTo = (lag+1):nsub2;
                    if ~isempty(rowsFrom)
                        sXsub(rowsTo, (k-1)*numel(baseCols) + (1:numel(baseCols))) = table2array(subTbl2(rowsFrom, baseCols));
                    end
                end
            end
            valid2 = all(~isnan(sXsub),2) & ~isnan(subTbl2.(targetCol));
            Xtrain = [Xtrain; sXsub(valid2, :)]; %#ok<AGROW>
            ytrain = [ytrain; subTbl2.(targetCol)(valid2)]; %#ok<AGROW>
        end
        if isempty(Xtrain), continue; end
        Xtrain_std = (Xtrain - muX) ./ sX;
        % lambda search (smaller Kfold to save time)
        mseGrid2 = nan(numel(lambdaGrid),1);
        for li = 1:numel(lambdaGrid)
            lam = lambdaGrid(li);
            try
                cvM2 = fitrlinear(Xtrain_std, ytrain, 'Learner','leastsquares','Regularization','ridge',...
                    'Lambda',lam,'KFold',3,'FitBias',true);
                mseGrid2(li) = kfoldLoss(cvM2,'LossFun','mse');
            catch
                mseGrid2(li) = NaN;
            end
        end
        [~, idxBest2] = min(mseGrid2);
        lamBest2 = lambdaGrid(idxBest2);
        mdlLOSO = fitrlinear(Xtrain_std, ytrain, 'Learner','leastsquares','Regularization','ridge',...
            'Lambda',lamBest2,'FitBias',true);
        % prepare test rows for subject s
        idxSub = find(ismember(T2.(subjectCol), s));
        if hasEpoch, [~,ord] = sort(T2.(epochCol)(idxSub)); idxSub = idxSub(ord); end
        nSub = numel(idxSub);
        subX = nan(nSub, numel(baseCols)*numel(lags));
        for k = 1:numel(lags)
            lag = lags(k);
            if lag == 0
                subX(:, (k-1)*numel(baseCols) + (1:numel(baseCols))) = table2array(T2(idxSub, baseCols));
            else
                rowsFrom = 1:(nSub - lag);
                rowsTo = (lag+1):nSub;
                if ~isempty(rowsFrom)
                    subX(rowsTo, (k-1)*numel(baseCols) + (1:numel(baseCols))) = table2array(T2(idxSub(rowsFrom), baseCols));
                end
            end
        end
        validMask = all(~isnan(subX),2) & ~isnan(T2.(targetCol)(idxSub));
        if ~any(validMask), continue; end
        subX_valid = subX(validMask, :);
        Xtest_std = (subX_valid - muX) ./ sX;
        ytest = T2.(targetCol)(idxSub);
        ytest = ytest(validMask);
        ypred = predict(mdlLOSO, Xtest_std);
        loso_preds = [loso_preds; ypred]; %#ok<AGROW>
        loso_truth = [loso_truth; ytest]; %#ok<AGROW>
    end
    if ~isempty(loso_preds)
        loso_rmse = sqrt(mean((loso_truth - loso_preds).^2));
        fprintf('LOSO RMSE (across subjects) = %.4f kg\n', loso_rmse);
    else
        fprintf('No LOSO predictions made (no valid test rows).\n');
    end
end

%% -------- Predict a 3.2 kg synthesized example --------
target_intermediate = 3.2;
mask25 = T2.(targetCol) == 2.5;
mask45 = T2.(targetCol) == 4.5;
if sum(mask25) < 1 || sum(mask45) < 1
    warning('Not enough 2.5 or 4.5 kg samples. Using global mean across available weights.');
end
mean_base_25 = mean(table2array(T2(mask25, baseCols)), 1, 'omitnan');
mean_base_45 = mean(table2array(T2(mask45, baseCols)), 1, 'omitnan');
if any(~isfinite(mean_base_25)), mean_base_25 = mean(table2array(T2(:, baseCols)),1,'omitnan'); end
if any(~isfinite(mean_base_45)), mean_base_45 = mean(table2array(T2(:, baseCols)),1,'omitnan'); end
alpha = (target_intermediate - 2.5) / (4.5 - 2.5);
base_interp = (1-alpha) * mean_base_25 + alpha * mean_base_45;
nLags = numel(lags);
x_lagged = repmat(base_interp, 1, nLags);
x_std = (x_lagged - muX) ./ sX;
if isfield(models, 'GLOBAL')
    mdlUse = models.GLOBAL;
    featureNamesUsed = results.GLOBAL.featureNames;
else
    fn = fieldnames(models); mdlUse = models.(fn{1}); featureNamesUsed = results.(fn{1}).featureNames;
end
maskUsed = ismember(featNames, featureNamesUsed);
x_for_model = x_std(maskUsed);
y_pred_interpolated = predict(mdlUse, x_for_model);
fprintf('Synthesized test weight %.2f kg -> predicted = %.3f kg\n', target_intermediate, y_pred_interpolated);

%% -------- Predict multiple random weights and tabulate errors --------
rng(randSeedTrials);
allWeights = T2.(targetCol);
uniqueWeightsAll = unique(allWeights);
nFeat = numel(baseCols);
meansPerW_raw = zeros(numel(uniqueWeightsAll), nFeat);
for wi = 1:numel(uniqueWeightsAll)
    wv = uniqueWeightsAll(wi);
    mask = allWeights == wv;
    meansPerW_raw(wi, :) = mean(table2array(T2(mask, baseCols)), 1, 'omitnan');
end
finiteRows = all(isfinite(meansPerW_raw), 2) & isfinite(uniqueWeightsAll);
uniqueWeights = uniqueWeightsAll(finiteRows);
meansPerW = meansPerW_raw(finiteRows, :);
fprintf('Found %d distinct valid weight levels (after removing NaNs/Infs).\n', numel(uniqueWeights));
if isempty(uniqueWeights), error('No valid weight levels.'); end
useNearestOnly = numel(uniqueWeights) == 1;

% choose model again
if isfield(models, 'GLOBAL')
    mdlUse = models.GLOBAL; featureNamesUsed = results.GLOBAL.featureNames;
else
    fn = fieldnames(models); mdlUse = models.(fn{1}); featureNamesUsed = results.(fn{1}).featureNames;
end
maskUsed = ismember(featNames, featureNamesUsed);

trialID = (1:numTrials).';
targetWeights = zeros(numTrials,1);
predictedWeights = zeros(numTrials,1);
signedError = zeros(numTrials,1);
absError = zeros(numTrials,1);
[uw_sorted, sortI] = sort(uniqueWeights);
V_sorted = meansPerW(sortI, :);

for t = 1:numTrials
    wt = randRangeMax * rand();
    targetWeights(t) = wt;
    if useNearestOnly
        base_interp = meansPerW(1, :);
    else
        try
            base_interp = interp1(uw_sorted, V_sorted, wt, 'linear', 'extrap');
            if any(~isfinite(base_interp)), error('interp1 returned non-finite'); end
        catch
            [~, idxMin] = min(abs(uw_sorted - wt));
            base_interp = V_sorted(idxMin, :);
            fprintf('Warning: interpolation failed for wt=%.3f. Using nearest %.3f\n', wt, uw_sorted(idxMin));
        end
    end
    x_lagged = repmat(base_interp, 1, nLags);
    x_std = (x_lagged - muX) ./ sX;
    x_for_model = x_std(maskUsed);
    y_pred = predict(mdlUse, x_for_model);
    predictedWeights(t) = y_pred;
    signedError(t) = y_pred - wt;
    absError(t) = abs(signedError(t));
end

resultsTable = table(trialID, targetWeights, predictedWeights, signedError, absError, ...
    'VariableNames', {'Trial', 'TargetWeight_kg', 'PredictedWeight_kg', 'SignedError_kg', 'AbsError_kg'});
disp(resultsTable);
fprintf('Mean absolute error over %d trials = %.4f kg\n', numTrials, mean(absError));

%% -------- Save everything --------
save(saveFile, 'models', 'results', 'muX', 'sX', 'lags', 'featNames', 'baseCols', 'subjects', ...
    'G', 'target_intermediate', 'y_pred_interpolated', 'resultsTable', 'targetWeights', 'predictedWeights', 'signedError', 'absError', '-v7.3');
fprintf('Saved models, preprocessing, and trial results to %s\n', saveFile);

%% -------- End of script --------

