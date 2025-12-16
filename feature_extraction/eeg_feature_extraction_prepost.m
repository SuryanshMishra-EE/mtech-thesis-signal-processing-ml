% ---------------------------------------------------------
% EEG Feature Extraction (Pre/Post Movement)
%
% This script extracts frequency-domain, time-domain, and
% Hjorth features from preprocessed EEG epochs.
%
% Input  : Preprocessed EEG (.mat files)
% Output : Feature table (MAT + CSV)
% ---------------------------------------------------------

clear; clc;

% 9) feature extraction (pre and post)
    for e = 1:EEGep.trials
        rowCounter = rowCounter + 1;
        srate = EEGep.srate;

        % find event sample (closest to 0 ms)
        [~, samp0] = min(abs(EEGep.times - 0));

        % PRE segment indices
        preStart = samp0 + round(preMoveWindow(1)*srate);
        preEnd   = samp0 + round(preMoveWindow(2)*srate);
        preStart = max(1, preStart);
        preEnd   = min(size(EEGep.data,2), preEnd);
        seg_pre = double(EEGep.data(:, preStart:preEnd)); % ch x time

        % POST segment indices
        postStart = samp0 + round(postMoveWindow(1)*srate);
        postEnd   = samp0 + round(postMoveWindow(2)*srate);
        postStart = max(1, postStart);
        postEnd   = min(size(EEGep.data,2), postEnd);
        seg_post = double(EEGep.data(:, postStart:postEnd)); % ch x time

        nch = size(seg_pre,1);

        % --- bandpower for pre and post ---
        bp_vals_pre = zeros(size(bands,1),1);
        bp_vals_post = zeros(size(bands,1),1);
        for b=1:size(bands,1)
            Pchan_pre = zeros(nch,1);
            Pchan_post = zeros(nch,1);
            for ch=1:nch
                % PRE
                sig = seg_pre(ch,:);
                siglen = length(sig);
                if siglen < 2 || all(sig==0)
                    Pchan_pre(ch) = 0;
                else
                    nfft = max(256, 2^nextpow2(siglen));
                    win  = min(round(2*srate), siglen);
                    noverlap = max(0, round(0.5*win));
                    if noverlap >= win, noverlap = max(0, win-1); end
                    try
                        [Pxx,F] = pwelch(sig, win, noverlap, nfft, srate);
                    catch
                        X = fft(sig, nfft);
                        Pxx = (abs(X).^2)/nfft;
                        F = (0:nfft-1)*(srate/nfft);
                    end
                    idx = F >= bands(b,1) & F <= bands(b,2);
                    if any(idx)
                        Pchan_pre(ch) = trapz(F(idx), Pxx(idx));
                    else
                        Pchan_pre(ch) = 0;
                    end
                end

                % POST
                sig2 = seg_post(ch,:);
                siglen2 = length(sig2);
                if siglen2 < 2 || all(sig2==0)
                    Pchan_post(ch) = 0;
                else
                    nfft2 = max(256, 2^nextpow2(siglen2));
                    win2  = min(round(2*srate), siglen2);
                    noverlap2 = max(0, round(0.5*win2));
                    if noverlap2 >= win2, noverlap2 = max(0, win2-1); end
                    try
                        [Pxx2,F2] = pwelch(sig2, win2, noverlap2, nfft2, srate);
                    catch
                        X2 = fft(sig2, nfft2);
                        Pxx2 = (abs(X2).^2)/nfft2;
                        F2 = (0:nfft2-1)*(srate/nfft2);
                    end
                    idx2 = F2 >= bands(b,1) & F2 <= bands(b,2);
                    if any(idx2)
                        Pchan_post(ch) = trapz(F2(idx2), Pxx2(idx2));
                    else
                        Pchan_post(ch) = 0;
                    end
                end
            end
            bp_vals_pre(b) = mean(Pchan_pre);
            bp_vals_post(b) = mean(Pchan_post);
        end

        % --- time stats pre (added kurtosis) ---
        allsig_pre = seg_pre(:);
        ft_mean_pre = mean(allsig_pre);
        ft_var_pre  = var(allsig_pre);
        ft_rms_pre  = sqrt(mean(allsig_pre.^2));
        ft_skew_pre = skewness(allsig_pre);
        ft_kurt_pre = kurtosis(allsig_pre);

        % --- time stats post (added kurtosis) ---
        allsig_post = seg_post(:);
        ft_mean_post = mean(allsig_post);
        ft_var_post  = var(allsig_post);
        ft_rms_post  = sqrt(mean(allsig_post.^2));
        ft_skew_post = skewness(allsig_post);
        ft_kurt_post = kurtosis(allsig_post);

        % --- Hjorth PRE ---
        hj_mob_pre = zeros(nch,1); hj_comp_pre = zeros(nch,1);
        for ch=1:nch
            x = seg_pre(ch,:);
            if numel(x) < 3
                hj_mob_pre(ch) = 0; hj_comp_pre(ch) = 0; continue;
            end
            dx = diff(x); ddx = diff(dx);
            v0 = var(x); v1 = var(dx); v2 = var(ddx);
            if v0 > 0 && v1 >= 0
                hj_mob_pre(ch) = sqrt(max(0,v1)/v0);
            else
                hj_mob_pre(ch) = 0;
            end
            if v1 > 0 && v2 >= 0
                hj_comp_pre(ch) = sqrt(max(0,v2)/v1);
            else
                hj_comp_pre(ch) = 0;
            end
        end
        hj_m_pre = mean(hj_mob_pre);
        hj_c_pre = mean(hj_comp_pre);

        % --- Hjorth POST ---
        hj_mob_post = zeros(nch,1); hj_comp_post = zeros(nch,1);
        for ch=1:nch
            x = seg_post(ch,:);
            if numel(x) < 3
                hj_mob_post(ch) = 0; hj_comp_post(ch) = 0; continue;
            end
            dx = diff(x); ddx = diff(dx);
            v0 = var(x); v1 = var(dx); v2 = var(ddx);
            if v0 > 0 && v1 >= 0
                hj_mob_post(ch) = sqrt(max(0,v1)/v0);
            else
                hj_mob_post(ch) = 0;
            end
            if v1 > 0 && v2 >= 0
                hj_comp_post(ch) = sqrt(max(0,v2)/v1);
            else
                hj_comp_post(ch) = 0;
            end
        end
        hj_m_post = mean(hj_mob_post);
        hj_c_post = mean(hj_comp_post);

        % table row (including kurtosis) - use normalized subj
        file_col    = string(fname);
        subj_col    = string(subj);
        weight_col  = double(weight);
        epoch_col   = double(e);

        % create band tables with prefix
        bp_pre_tbl = array2table(bp_vals_pre', 'VariableNames', strcat("pre_", bandNames));
        bp_post_tbl = array2table(bp_vals_post', 'VariableNames', strcat("post_", bandNames));

        Tsmall = table(file_col, subj_col, weight_col, epoch_col, ...
            ft_mean_pre, ft_var_pre, ft_rms_pre, ft_skew_pre, ft_kurt_pre, hj_m_pre, hj_c_pre, ...
            ft_mean_post, ft_var_post, ft_rms_post, ft_skew_post, ft_kurt_post, hj_m_post, hj_c_post, ...
            'VariableNames', {'file','subject','weight','epoch', ...
            'mean_pre','var_pre','rms_pre','skew_pre','kurt_pre','hj_mob_pre','hj_comp_pre', ...
            'mean_post','var_post','rms_post','skew_post','kurt_post','hj_mob_post','hj_comp_post'});

        Trow = [Tsmall bp_pre_tbl bp_post_tbl];
        FeatTable = [FeatTable; Trow]; %#ok<AGROW>
    end

    % save processed file
    outname = fullfile(outFolder, ['preproc_' fname(1:end-5) '.mat']);
    try
        save(outname,'EEG','EEGep','EEGica','badChNames','eventMarkers','epochWindow','preMoveWindow','postMoveWindow','-v7.3');
        fprintf('  Saved: %s | epochs kept: %d\n', outname, EEGep.trials);
    catch
        warning('  Save failed for %s', fname);
    end
end

% save feature table
featFile = fullfile(outFolder,'features_prePost_withnotch_kurtosis_normalized.mat');
save(featFile,'FeatTable','bands','bandNames','-v7.3');

finalCsv = fullfile(outFolder,'finalcsv_withkurtosis_normalized.csv');
writetable(FeatTable, finalCsv);

% --- save interpolation summary CSV ---
interpCsv = fullfile(outFolder,'interp_summary_normalized.csv');
if ~isempty(interpSummary)
    writetable(interpSummary, interpCsv);
    fprintf('\nInterpolation summary saved: %s\n', interpCsv);
end

% print overall interpolation stats
if ~isempty(interpPct_all)
    fprintf('\nInterpolation summary across files:\n');
    fprintf('  Mean: %.2f%%\n', mean(interpPct_all));
    fprintf('  Median: %.2f%%\n', median(interpPct_all));
    fprintf('  Max: %.2f%%\n', max(interpPct_all));
    fprintf('  Min: %.2f%%\n', min(interpPct_all));
else
    fprintf('\nNo channels were flagged for interpolation in this run.\n');
end


fprintf('\nPreprocessing complete. Outputs in: %s\nSaved feature CSV: %s\n', outFolder, finalCsv);
