function validate_official_olsa_sub03()
subject = 'sub-03';
repo_root = 'E:\decode\_meg_scans_official_preprocessing_sub03_v1';
local_derivatives = 'E:\decode\data\derived\meg_scans_official_replication_v1';
out_dir = fullfile(repo_root, 'experiments', 'meg_scans_official_preprocessing_sub03_v1');
olsa_path = fullfile(local_derivatives, subject, 'speech', 'sub-03_preprocessed_olsa_decoding.mat');
audiobook_path = fullfile(local_derivatives, subject, 'speech', 'sub-03_preprocessed_audiobooks_decoding.mat');

olsa_loaded = load(olsa_path, 'results');
olsa = olsa_loaded.results;
audiobook_loaded = load(audiobook_path, 'results');
audiobook = audiobook_loaded.results;

required = {'settings', 'epochs_neuro', 'epochs_audio', 'event_description'};
for i = 1:numel(required)
    if ~isfield(olsa, required{i})
        error('Missing OLSA results.%s', required{i});
    end
end

if numel(olsa.epochs_neuro.trial) ~= 120 || numel(olsa.epochs_audio) ~= 120
    error('Expected exactly 120 OLSA trials.');
end
if numel(olsa.event_description.snrs) ~= 120 || numel(olsa.event_description.intells) ~= 120 || numel(olsa.event_description.playlist) ~= 120
    error('Expected 120 SNR/intelligibility/playlist entries.');
end
if ~isequal(olsa.epochs_neuro.label, audiobook.epochs_neuro.label)
    error('OLSA and audiobook MEG channel labels/order differ.');
end
if olsa.epochs_neuro.fsample ~= 64
    error('Unexpected OLSA fsample.');
end

settings = olsa.settings;
assert(settings.use_maxfilter == true);
assert(settings.apply_latency_correction == true);
assert(abs(settings.audio_latency - 0.003) < 1e-12);
assert(isequal(settings.decoding.bpfreq, [0.5, 4]));
assert(settings.decoding.fs_neuro == 1000);
assert(settings.decoding.fs_down == 64);
assert(settings.decoding.olsa.prestim == 0);
assert(settings.decoding.olsa.poststim == 0.5);

min_len = inf;
max_len = -inf;
for trl_idx = 1:120
    neuro = olsa.epochs_neuro.trial{trl_idx};
    audio = olsa.epochs_audio{trl_idx};
    if size(neuro, 1) ~= 306
        error('Unexpected channel count at OLSA trial %d.', trl_idx);
    end
    if size(neuro, 2) ~= numel(audio)
        error('OLSA neuro/audio length mismatch at trial %d.', trl_idx);
    end
    min_len = min(min_len, size(neuro, 2));
    max_len = max(max_len, size(neuro, 2));
end

summary = struct;
summary.output_mat = olsa_path;
summary.validation_status = 'passed';
summary.subject = subject;
summary.n_trials = 120;
summary.channel_count = 306;
summary.fsample_hz = olsa.epochs_neuro.fsample;
summary.min_trial_len_samples = min_len;
summary.max_trial_len_samples = max_len;
summary.has_120_snr = true;
summary.has_120_intelligibility = true;
summary.has_120_playlist = true;
summary.channel_order_matches_audiobook = true;
summary.use_maxfilter = settings.use_maxfilter;
summary.audio_latency_seconds = settings.audio_latency;
summary.bandpass_hz = settings.decoding.bpfreq;
summary.fs_neuro_hz = settings.decoding.fs_neuro;
summary.fs_down_hz = settings.decoding.fs_down;
summary.olsa_prestim_seconds = settings.decoding.olsa.prestim;
summary.olsa_poststim_seconds = settings.decoding.olsa.poststim;

fid = fopen(fullfile(out_dir, 'sub03_official_olsa_validation_summary.json'), 'w');
fprintf(fid, '%s', jsonencode(summary, 'PrettyPrint', true));
fclose(fid);

rows = cell(120, 7);
for trl_idx = 1:120
    rows(trl_idx, :) = {subject, trl_idx, size(olsa.epochs_neuro.trial{trl_idx}, 1), size(olsa.epochs_neuro.trial{trl_idx}, 2), numel(olsa.epochs_audio{trl_idx}), olsa.event_description.snrs(trl_idx), olsa.event_description.intells(trl_idx)};
end
T = cell2table(rows, 'VariableNames', {'subject_id', 'trial_index', 'meg_channels', 'neuro_len', 'audio_len', 'SNR', 'intelligibility'});
writetable(T, fullfile(out_dir, 'sub03_official_olsa_trial_validation.csv'));

fprintf('official OLSA validation passed\n');
end
