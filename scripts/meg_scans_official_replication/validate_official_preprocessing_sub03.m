function validate_official_preprocessing_sub03()
subject = 'sub-03';
repo_root = 'E:\decode\_meg_scans_official_preprocessing_sub03_v1';
official_repo = 'E:\decode\external\upstream\MEG-SCANS';
dataset_root = 'E:\decode\data\raw\meg_scans_ds006468';
local_derivatives = 'E:\decode\data\derived\meg_scans_official_replication_v1';
out_dir = fullfile(repo_root, 'experiments', 'meg_scans_official_preprocessing_sub03_v1');
mat_path = fullfile(local_derivatives, subject, 'speech', 'sub-03_preprocessed_audiobooks_decoding.mat');

fieldtrip_path = 'E:\decode\external\toolboxes\fieldtrip-master\fieldtrip-master';
addpath(fieldtrip_path);
ft_defaults;
addpath(fullfile(official_repo, 'helper_functions'));

if ~exist(mat_path, 'file')
    error('Output mat not found: %s', mat_path);
end
loaded = load(mat_path, 'results');
results = loaded.results;
required = {'epochs_neuro', 'epochs_audio', 'mapping_epochs', 'mapping_label'};
for i = 1:numel(required)
    if ~isfield(results, required{i})
        error('Missing results.%s', required{i});
    end
end

expected_labels = {'task-audiobook1_run-01', 'task-audiobook1_run-02', 'task-audiobook2_run-01', 'task-audiobook2_run-02'};
actual_labels = results.mapping_label;
if ~isequal(actual_labels, expected_labels)
    error('Mapping label mismatch.');
end

audio_mat = fullfile(dataset_root, 'derivatives', 'stimuli', 'sub-others_preprocessed_audiobook_envelopes_decoding.mat');
audio = importdata(audio_mat);
if ~isequal(audio.audiobook_labels, expected_labels)
    error('Official audio labels mismatch.');
end

settings = results.settings;
assert(settings.use_maxfilter == true);
assert(settings.apply_latency_correction == true);
assert(abs(settings.audio_latency - 0.003) < 1e-12);
assert(isequal(settings.decoding.bpfreq, [0.5, 4]));
assert(settings.decoding.trialdur == 120);
assert(settings.decoding.fs_neuro == 1000);
assert(settings.decoding.fs_down == 64);
assert(results.epochs_neuro.fsample == 64);

if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

summary = struct;
summary.output_mat = mat_path;
summary.validation_status = 'passed';
summary.subject = subject;
summary.mapping_labels = actual_labels;
summary.fs_down_hz = results.epochs_neuro.fsample;
summary.trialdur_seconds = settings.decoding.trialdur;
summary.bandpass_hz = settings.decoding.bpfreq;
summary.audio_latency_seconds = settings.audio_latency;
summary.use_maxfilter = settings.use_maxfilter;
summary.total_trials = numel(results.epochs_audio);
summary.channel_count = numel(results.epochs_neuro.label);
summary.min_dur_samples = numel(results.epochs_audio{1});
summary.min_dur_seconds = summary.min_dur_samples / results.epochs_neuro.fsample;

rows = {};
for f_idx = 1:numel(expected_labels)
    label = expected_labels{f_idx};
    idx_start = results.mapping_epochs(f_idx, 1);
    idx_end = results.mapping_epochs(f_idx, 2);
    trial_idx = idx_start:idx_end;
    n_kept = numel(trial_idx);
    n_audio_original = numel(audio.epochs_audio{f_idx}.trial);
    n_neuro_defined = count_defined_neuro_trials(dataset_root, subject, label, settings);
    n_removed_total = n_neuro_defined - n_kept;
    n_extra_audio_removed = n_audio_original - n_neuro_defined;

    if n_kept <= 0
        error('Empty retained trial set for %s', label);
    end

    min_len = inf;
    max_len = -inf;
    min_audio_len = inf;
    max_audio_len = -inf;
    for k = trial_idx
        neuro_len = size(results.epochs_neuro.trial{k}, 2);
        audio_len = numel(results.epochs_audio{k});
        if neuro_len ~= audio_len
            error('Length mismatch after truncation for %s trial %d', label, k);
        end
        if size(results.epochs_neuro.trial{k}, 1) ~= 306
            error('Unexpected MEG channel count for %s trial %d', label, k);
        end
        min_len = min(min_len, neuro_len);
        max_len = max(max_len, neuro_len);
        min_audio_len = min(min_audio_len, audio_len);
        max_audio_len = max(max_audio_len, audio_len);
    end

    reason = 'none';
    if n_extra_audio_removed > 0 && n_removed_total == 0
        reason = 'extra_audio_trials_removed_by_official_script';
    elseif n_removed_total > 0
        reason = 'short_neuro_trials_removed_by_official_script';
    end

    rows(end+1, :) = {subject, label, n_neuro_defined, n_audio_original, n_kept, n_extra_audio_removed, n_removed_total, min_len, max_len, min_audio_len, max_audio_len, reason}; %#ok<AGROW>
end

T = cell2table(rows, 'VariableNames', {'subject_id', 'mapping_label', 'neuro_trials_defined', 'audio_trials_original', 'paired_trials_kept', 'extra_audio_trials_removed', 'neuro_trials_removed', 'min_neuro_len', 'max_neuro_len', 'min_audio_len', 'max_audio_len', 'removal_reason'});
writetable(T, fullfile(out_dir, 'sub03_official_trial_validation.csv'));

fid = fopen(fullfile(out_dir, 'sub03_official_validation_summary.json'), 'w');
fprintf(fid, '%s', jsonencode(summary, 'PrettyPrint', true));
fclose(fid);

fid = fopen(fullfile(out_dir, 'sub03_official_preprocessing_log_summary.md'), 'w');
fprintf(fid, '# sub-03 Official Preprocessing Log Summary\n\n');
fprintf(fid, '- Output MAT: `%s`\n', mat_path);
fprintf(fid, '- Validation status: passed\n');
fprintf(fid, '- Total paired trials: %d\n', summary.total_trials);
fprintf(fid, '- MEG fsample: %g Hz\n', summary.fs_down_hz);
fprintf(fid, '- MEG channel count: %d\n', summary.channel_count);
fprintf(fid, '- Trial duration protocol: %g s\n', summary.trialdur_seconds);
fprintf(fid, '- Final truncation length: %d samples (%.6f s)\n', summary.min_dur_samples, summary.min_dur_seconds);
fprintf(fid, '- Mapping labels: `%s`\n', strjoin(actual_labels, '`, `'));
fclose(fid);

fprintf('official preprocessing validation passed\n');
end

function n = count_defined_neuro_trials(dataset_root, subject, label, settings)
parts = split(label, '_');
task = erase(parts{1}, 'task-');
run = erase(parts{2}, 'run-');
event_path = fullfile(dataset_root, subject, 'meg', sprintf('%s_task-%s_run-%s_meg.fif', subject, task, run));
cfg = [];
cfg.trialfun = 'my_trialfun_audiobook';
cfg.dataset = event_path;
cfg.trialdef.method = 'fieldtrip';
cfg.trialdef.trialdur = settings.decoding.trialdur;
cfg.trialdef.eventvalue = 1;
cfg.trialdef.eventtype = 'STI101';
cfg = ft_definetrial(cfg);
n = size(cfg.trl, 1);
end
