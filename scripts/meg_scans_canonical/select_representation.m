function [selected_trials, selected_labels, source_indices_1based] = select_representation(epochs_neuro, representation, selection_json_path)
% Select canonical MEG-SCANS sensor representations with channel-order checks.
if nargin < 3 || isempty(selection_json_path)
    selection_json_path = fullfile('E:\decode\_meg_scans_canonical_05_08hz_64hz_sub03_v1', ...
        'experiments', 'meg_scans_canonical_05_08hz_64hz_sub03_v1', 'mag102_selection.json');
end

labels = epochs_neuro.label(:);
switch lower(representation)
    case 'all306'
        source_indices_1based = 1:numel(labels);
        selected_labels = labels;
    case 'mag102'
        if ~exist(selection_json_path, 'file')
            error('Missing mag102 selection JSON: %s', selection_json_path);
        end
        selection = jsondecode(fileread(selection_json_path));
        source_indices_1based = double(selection.source_indices_1based(:))';
        selected_labels = cellstr(selection.ordered_channel_labels(:));
        observed_labels = labels(source_indices_1based);
        if ~isequal(observed_labels(:), selected_labels(:))
            error('mag102 channel labels do not match mag102_selection.json.');
        end
        observed_hash = sha256_text(strjoin(observed_labels, newline));
        if ~strcmp(observed_hash, selection.channel_order_sha256)
            error('mag102 channel-order hash mismatch. Expected %s, observed %s.', selection.channel_order_sha256, observed_hash);
        end
    otherwise
        error('Unsupported representation: %s. Expected all306 or mag102.', representation);
end

selected_trials = cell(size(epochs_neuro.trial));
for idx = 1:numel(epochs_neuro.trial)
    trial = epochs_neuro.trial{idx};
    if size(trial, 1) ~= numel(labels)
        error('Trial %d channel count does not match label metadata.', idx);
    end
    selected_trials{idx} = trial(source_indices_1based, :);
end
end

function hash = sha256_text(text)
md = java.security.MessageDigest.getInstance('SHA-256');
bytes = uint8(text);
md.update(bytes);
hash = lower(reshape(dec2hex(typecast(md.digest(), 'uint8'))', 1, []));
end
