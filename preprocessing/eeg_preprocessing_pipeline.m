clear; clc;

dataFolder = 'C:\Users\Suryansh Mishra\Documents\Thesis\Task_Data-20250921T054507Z-1-001\Task_Data';
outFolder  = fullfile(dataFolder,'preprocessed_full');
if ~exist(outFolder,'dir'), mkdir(outFolder); end

eventMarkers = {'L','U','M1'};   % markers to epoch around
epochWindow   = [-1.0 1.5];      % seconds around marker
preMoveWindow = [-1.0 0];        % pre-movement window relative to marker (s)
postMoveWindow = [0 1.0];        % post-movement window relative to marker (s)

hp_cut = 0.5; lp_cut = 100;      % bandpass limits (Hz) - lowpass now 100 Hz
notchFreq = 50; notchBW = 2;     % mains notch
targetFs  = 250;                 % resample target (Hz)

epoch_rej_thresh_uV = 300;       % rejection threshold (uV)
badchan_std_ratio = 5;           % bad channel detection

bands = [0.5 4; 4 8; 8 13; 13 30; 30 45];
bandNames = {'delta','theta','alpha','beta','lowgamma'};
% -------------------------------------------------

% check EEGLAB
if exist('eeglab','file')==2
    eeglab('nogui'); close(gcf);
else
    error('EEGLAB not on MATLAB path.');
end
if exist('pop_loadbv','file')~=2
    error('pop_loadbv not found. Install bva-io plugin.');
end

vhdrFiles = dir(fullfile(dataFolder,'*.vhdr'));
if isempty(vhdrFiles)
    error('No .vhdr files found in %s', dataFolder);
end

FeatTable = table();
rowCounter = 0;

% --- interpolation tracking containers ---
interpPct_all = [];                % percentage per file
interpSummary = table();           % filewise summary table

for f = 1:numel(vhdrFiles)
    fname = vhdrFiles(f).name;
    fprintf('\n[%d/%d] %s\n', f, numel(vhdrFiles), fname);
    try
        EEGraw = pop_loadbv(dataFolder, fname);
        EEGraw = eeg_checkset(EEGraw);
    catch ME
        warning('Load failed %s : %s', fname, ME.message); continue;
    end

    % parse subject and weight
    parts = strsplit(fname,'_');
    subj_raw = parts{1};
    % normalize known misspellings
    if strcmpi(subj_raw,'Hrshal')
        subj = 'Harshal';
    else
        subj = subj_raw;
    end

    wtTok = regexp(fname,'(\d+\.?\d*)\s*kg','tokens','once');
    if isempty(wtTok), wtTok = regexp(fname,'[_\- ](\d+\.?\d*)(?=\.)','tokens','once'); end
    if ~isempty(wtTok), weight = str2double(wtTok{1}); else weight = NaN; end

    has_chanlocs = isfield(EEGraw,'chanlocs') && ~isempty({EEGraw.chanlocs.labels});

    % record original channel count before any removal
    orig_n = EEGraw.nbchan;

    % 1) filtering: HP -> notch -> LP
    EEG = EEGraw;
    try
        EEG = pop_eegfiltnew(EEG, hp_cut, [], [], 0);
        EEG = pop_eegfiltnew(EEG, notchFreq - notchBW/2, notchFreq + notchBW/2, [], 1);
        EEG = pop_eegfiltnew(EEG, [], lp_cut, [], 0);
    catch
        warning('Filtering failed for %s.', fname);
    end

    % 2) resample
    if ~isempty(targetFs) && EEG.srate ~= targetFs
        try
            EEG = pop_resample(EEG, targetFs);
        catch
            warning('Resample failed for %s. Keeping original srate.', fname);
        end
    end

    % 3) bad channel detection
    datAll = double(reshape(EEG.data, EEG.nbchan, []));
    chVar = var(datAll,0,2);
    medVar = median(chVar);
    badChIdx = find(chVar > badchan_std_ratio * medVar | chVar == 0);
    badChNames = {};
    if ~isempty(badChIdx)
        if has_chanlocs
            badChNames = {EEG.chanlocs(badChIdx).labels};
            fprintf('  Bad channels: %s\n', strjoin(badChNames,','));
        else
            fprintf('  Bad channel indices: %s\n', mat2str(badChIdx));
        end
        try
            EEG = pop_select(EEG,'nochannel',badChIdx);
        catch
            warning('  Channel removal failed.');
        end
    end

    % record interpolation info for this file (will be interpolated later if chanlocs exist)
    n_interp = numel(badChIdx);
    pct_interp = 100 * n_interp / orig_n;
    interpPct_all(end+1) = pct_interp; %#ok<SAGROW>
    % append to table (use normalized subj)
    Ttmp = table(string(fname), orig_n, n_interp, pct_interp, string(subj), 'VariableNames', ...
        {'file','orig_channels','n_interpolated','pct_interpolated','subject'});
    interpSummary = [interpSummary; Ttmp]; %#ok<AGROW>

    fprintf('  Interpolated %d of %d channels = %.1f%%\n', n_interp, orig_n, pct_interp);

    % 4) re-reference
    try
        if EEG.nbchan > 1
            EEG = pop_reref(EEG, []);
        end
    catch
        warning('  Reref failed.');
    end

    % 5) epoching
    try
        EEGep = pop_epoch(EEG, eventMarkers, epochWindow, 'epochinfo','yes');
        EEGep = eeg_checkset(EEGep);
        EEGep = pop_rmbase(EEGep, [epochWindow(1)*1000 0]);
    catch ME
        warning('  Epoching failed for %s : %s. Saving continuous only.', fname, ME.message);
        outname = fullfile(outFolder, ['preproc_cont_' fname(1:end-5) '.mat']);
        save(outname,'EEG','subj','weight','-v7.3');
        continue;
    end

    % 6) artifact rejection
    rejEpochs = [];
    for e=1:EEGep.trials
        dat = EEGep.data(:,:,e);
        p2p = max(dat,[],2) - min(dat,[],2);
        if any(p2p > epoch_rej_thresh_uV)
            rejEpochs(end+1) = e; %#ok<AGROW>
        end
    end
    if ~isempty(rejEpochs)
        if numel(rejEpochs) == EEGep.trials
            fprintf('  All %d epochs exceed threshold. Keeping them.\n', EEGep.trials);
        else
            fprintf('  Rejecting %d/%d epochs\n', numel(rejEpochs), EEGep.trials);
            EEGep = pop_rejepoch(EEGep, rejEpochs, 0);
        end
    end

    % 7) interpolate 
    if ~isempty(badChIdx) && has_chanlocs
        try
            EEGfull = pop_loadbv(dataFolder, fname); EEGfull = eeg_checkset(EEGfull);
            EEGep = pop_interp(EEGep, EEGfull.chanlocs, 'spherical');
            fprintf('  Interpolated removed channels.\n');
        catch
            warning('  Interpolation failed.');
        end
    end

    % 8) ICA 
    EEGica = [];
    try
        EEGica = EEGep;
        pcaDim = min(EEGica.nbchan-1, EEGica.nbchan);
        EEGica = pop_runica(EEGica, 'icatype','runica','extended',1,'pca',pcaDim);
    catch
        warning('  ICA failed for %s.', fname);
        EEGica = [];
    end