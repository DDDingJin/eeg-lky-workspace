function validate_canonical_sub03_05_08hz_64hz()
subject = 'sub-03';
repo_root = 'E:\decode\_meg_scans_canonical_05_08hz_64hz_sub03_v1';
dataset_root = 'E:\decode\data\raw\meg_scans_ds006468';
local_derivatives = 'E:\decode\data\derived\meg_scans_canonical_05_08hz_64hz_v1';
official_repo = 'E:\decode\external\upstream\MEG-SCANS';
expected_official_commit = '32bfc690e28e7591b45d96615c59b2d53b6a7165';
out_dir = fullfile(repo_root, 'experiments', 'meg_scans_canonical_05_08hz_64hz_sub03_v1');
paired_path = fullfile(local_derivatives, subject, 'speech', 'sub-03_preprocessed_audiobooks_decoding.mat');
envelope_path = fullfile(local_derivatives, 'stimuli', 'sub-others_preprocessed_audiobook_envelopes_decoding.mat');
official_envelope_path = fullfile(dataset_root, 'derivatives', 'stimuli', 'sub-others_preprocessed_audiobook_envelopes_decoding.mat');

fieldtrip_path = 'E:\decode\external\toolboxes\fieldtrip-master\fieldtrip-master';
addpath(fieldtrip_path);
ft_defaults;
addpath(fullfile(official_repo, 'helper_functions'));

if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end
if ~exist(paired_path, 'file')
    error('Missing canonical paired MAT: %s', paired_path);
end
if ~exist(envelope_path, 'file')
    error('Missing canonical envelope MAT: %s', envelope_path);
end

paired_loaded = load(paired_path, 'results');
results = paired_loaded.results;
env_loaded = load(envelope_path, 'audio_envelopes');
audio_envelopes = env_loaded.audio_envelopes;
official_env = load(official_envelope_path, 'audio_envelopes');
observed_official_commit = get_git_commit(official_repo);
upstream_check = struct;
upstream_check.expected_commit = expected_official_commit;
upstream_check.observed_commit = observed_official_commit;
upstream_check.repository_path = official_repo;
upstream_check.status = ternary(strcmp(observed_official_commit, expected_official_commit), 'passed', 'failed');
upstream_check.checked_at = char(datetime('now', 'TimeZone', 'local', 'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX'));
upstream_check.scope = 'post-hoc verification for already generated canonical data; not generation-time proof';
write_json(fullfile(out_dir, 'upstream_version_check.json'), upstream_check);
if ~strcmp(observed_official_commit, expected_official_commit)
    error('MEG-SCANS upstream commit mismatch. Expected %s, observed %s.', expected_official_commit, observed_official_commit);
end

expected_labels = {'task-audiobook1_run-01', 'task-audiobook1_run-02', 'task-audiobook2_run-01', 'task-audiobook2_run-02'};
if ~isequal(results.mapping_label, expected_labels)
    error('Canonical mapping labels mismatch.');
end
if ~isequal(audio_envelopes.audiobook_labels, expected_labels)
    error('Canonical envelope labels mismatch.');
end
if ~isequal(official_env.audio_envelopes.audiobook_labels, expected_labels)
    error('Official envelope labels mismatch.');
end

settings = results.settings;
assert(settings.use_maxfilter == true);
assert(settings.apply_latency_correction == true);
assert(abs(settings.audio_latency - 0.003) < 1e-12);
assert(isequal(settings.decoding.bpfreq, [0.5, 8]));
assert(settings.decoding.trialdur == 120);
assert(settings.decoding.fs_neuro == 1000);
assert(settings.decoding.fs_down == 64);
assert(results.epochs_neuro.fsample == 64);
assert(numel(results.epochs_audio) == 16);
assert(numel(results.epochs_neuro.trial) == 16);

[inventoryT, mag_selection] = build_channel_inventory(results.epochs_neuro, out_dir);

protocol = struct;
protocol.artifact = 'protocol_lock.json';
protocol.scope = 'MEG-SCANS canonical sub-03 audiobook preprocessing only; no training';
protocol.subject = subject;
protocol.base_branch = 'fix/ar-20260711-meg-scans-official-preprocessing-sub03-v1';
protocol.base_commit = '8e8c694ee3cca1121b2031b0d00ddcfc0395d6bf';
protocol.official_meg_scans_repo = official_repo;
protocol.official_meg_scans_commit = expected_official_commit;
protocol.observed_meg_scans_commit_posthoc = observed_official_commit;
protocol.meg_source = fullfile(dataset_root, 'derivatives', subject, 'maxfilter', '*_proc-tsss-mc_meg.fif');
protocol.wav_source = fullfile(dataset_root, 'stimuli', 'audiobooks', '*_stim.wav');
protocol.envelope_source = 'raw WAV only; not refiltered from official 0.5-4 Hz envelope';
protocol.envelope_method = 'AMT auditoryfilterbank flow=50 fhigh=5000 -> abs(.)^0.6 -> mean across bands';
protocol.temporal_band_hz = [0.5, 8];
protocol.filtertype = 'firws';
protocol.fs_audio_hz = 44100;
protocol.fs_neuro_hz = 1000;
protocol.fs_down_hz = 64;
protocol.trialdur_seconds = 120;
protocol.audio_latency_seconds = 0.003;
protocol.normalization = 'none during preprocessing; future scaler must fit on training split only';
protocol.local_envelope_mat = envelope_path;
protocol.local_paired_mat = paired_path;
write_json(fullfile(out_dir, 'protocol_lock.json'), protocol);

run_rows = {};
trial_rows = {};
psd_rows = {};
invalid_rows = {};
total_kept = 0;

for f_idx = 1:numel(expected_labels)
    label = expected_labels{f_idx};
    parts = split(label, '_');
    task = erase(parts{1}, 'task-');
    run = erase(parts{2}, 'run-');
    stem = sprintf('%s_task-%s_run-%s', subject, task, run);
    fif_path = fullfile(dataset_root, 'derivatives', subject, 'maxfilter', sprintf('%s_proc-tsss-mc_meg.fif', stem));
    wav_path = fullfile(dataset_root, 'stimuli', 'audiobooks', sprintf('task-%s_run-%s_stim.wav', task, run));
    event_path = fullfile(dataset_root, subject, 'meg', sprintf('%s_meg.fif', stem));
    cfg = [];
    cfg.trialfun = 'my_trialfun_audiobook';
    cfg.dataset = event_path;
    cfg.trialdef.method = 'fieldtrip';
    cfg.trialdef.trialdur = 120;
    cfg.trialdef.eventvalue = 1;
    cfg.trialdef.eventtype = 'STI101';
    cfg = ft_definetrial(cfg);
    n_defined = size(cfg.trl, 1);

    idx_start = results.mapping_epochs(f_idx, 1);
    idx_end = results.mapping_epochs(f_idx, 2);
    trial_idx = idx_start:idx_end;
    n_kept = numel(trial_idx);
    total_kept = total_kept + n_kept;
    n_audio_generated = numel(audio_envelopes.epochs_audio{f_idx}.trial);
    n_invalid = n_defined - n_kept;

    run_rows(end+1, :) = {subject, label, fif_path, exist(fif_path, 'file') == 2, wav_path, exist(wav_path, 'file') == 2, n_defined, n_audio_generated, n_kept, n_invalid}; %#ok<AGROW>
    if n_invalid > 0
        invalid_rows(end+1, :) = {subject, label, n_invalid, 'official short-trial cleanup: final neuro trial shorter than 99% of 120 s target'}; %#ok<AGROW>
    end

    for local_idx = 1:n_kept
        global_idx = trial_idx(local_idx);
        meg = results.epochs_neuro.trial{global_idx};
        env = results.epochs_audio{global_idx};
        if size(meg, 1) ~= 306
            error('Unexpected MEG channel count for %s trial %d.', label, local_idx);
        end
        if size(meg, 2) ~= 7680 || numel(env) ~= 7680
            error('Unexpected trial length for %s trial %d.', label, local_idx);
        end
        if size(meg, 2) ~= numel(env)
            error('MEG/envelope mismatch for %s trial %d.', label, local_idx);
        end
        trial_rows(end+1, :) = {subject, label, local_idx, global_idx, 306, size(meg, 2), numel(env), 64, 'valid'}; %#ok<AGROW>
    end

    env_trials = results.epochs_audio(trial_idx);
    meg_trials = results.epochs_neuro.trial(trial_idx);
    psd_rows = append_psd_rows(psd_rows, subject, label, 'envelope', env_trials, 64);
    psd_rows = append_psd_rows(psd_rows, subject, label, 'meg_mean_channels', cellfun(@(x) mean(x, 1), meg_trials, 'UniformOutput', false), 64);
end

if total_kept ~= 16
    error('Expected 16 valid paired trials, got %d.', total_kept);
end

runT = cell2table(run_rows, 'VariableNames', {'subject_id', 'run_label', 'meg_source_path', 'meg_source_exists', 'wav_source_path', 'wav_source_exists', 'neuro_trials_defined', 'audio_trials_generated', 'paired_trials_kept', 'invalid_trials'});
writetable(runT, fullfile(out_dir, 'run_manifest.csv'));

trialT = cell2table(trial_rows, 'VariableNames', {'subject_id', 'run_label', 'trial_index_in_run', 'global_trial_index', 'meg_channels', 'meg_samples', 'envelope_samples', 'fs_hz', 'validity'});
writetable(trialT, fullfile(out_dir, 'trial_manifest.csv'));

if isempty(invalid_rows)
    invalid_rows = {subject, 'none', 0, 'none'};
end
invalidT = cell2table(invalid_rows, 'VariableNames', {'subject_id', 'run_label', 'invalid_trials', 'reason'});
writetable(invalidT, fullfile(out_dir, 'invalid_trials.csv'));

psdT = cell2table(psd_rows, 'VariableNames', {'subject_id', 'run_label', 'signal', 'band_hz', 'absolute_power', 'relative_power_0p5_16'});
writetable(psdT, fullfile(out_dir, 'psd_summary.csv'));

channelwise_psdT = build_channelwise_psd(results, expected_labels, inventoryT);
writetable(channelwise_psdT, fullfile(out_dir, 'channelwise_psd_summary.csv'));

run_manifest = struct;
run_manifest.artifact = 'run_manifest.json';
run_manifest.subject = subject;
run_manifest.local_derivatives_root = local_derivatives;
run_manifest.local_envelope_mat = envelope_path;
run_manifest.local_paired_mat = paired_path;
run_manifest.mapping_labels = expected_labels;
run_manifest.valid_paired_trials = total_kept;
run_manifest.expected_samples_per_trial = 7680;
run_manifest.expected_meg_channels = 306;
run_manifest.neural_band_hz = [0.5, 8];
run_manifest.envelope_band_hz = [0.5, 8];
run_manifest.fs_hz = 64;
run_manifest.no_zscore = true;
run_manifest.official_anchor_0p5_4_untouched = true;
write_json(fullfile(out_dir, 'run_manifest.json'), run_manifest);

provenance = struct;
provenance.artifact = 'data_provenance.json';
provenance.scope = 'training-preflight metadata/provenance closure only';
provenance.local_paired_mat = file_record(paired_path);
provenance.local_envelope_mat = file_record(envelope_path);
provenance.upstream_version_check_path = fullfile(out_dir, 'upstream_version_check.json');
provenance.channel_inventory_path = fullfile(out_dir, 'channel_inventory.csv');
provenance.mag102_selection_path = fullfile(out_dir, 'mag102_selection.json');
provenance.channelwise_psd_summary_path = fullfile(out_dir, 'channelwise_psd_summary.csv');
provenance.preprocessing_rerun = false;
provenance.training_run = false;
provenance.data_body_modified = false;
provenance.note = 'Existing canonical paired/envelope MAT files were only read and hashed; no preprocessing or training was run.';
write_json(fullfile(out_dir, 'data_provenance.json'), provenance);

fid = fopen(fullfile(out_dir, 'validation_report.md'), 'w');
fprintf(fid, '# MEG-SCANS canonical 0.5-8 Hz / 64 Hz sub-03 validation\n\n');
fprintf(fid, '- Local envelope MAT: `%s`\n', envelope_path);
fprintf(fid, '- Local paired MAT: `%s`\n', paired_path);
fprintf(fid, '- Valid paired trials: `%d`\n', total_kept);
fprintf(fid, '- Samples per trial: `7680`\n');
fprintf(fid, '- MEG channels: `306`\n');
fprintf(fid, '- MEG band: `0.5-8 Hz`\n');
fprintf(fid, '- Envelope band: `0.5-8 Hz`\n');
fprintf(fid, '- Final fs: `64 Hz`\n');
fprintf(fid, '- Audio latency correction: `3 ms`\n');
fprintf(fid, '- Normalization: none in preprocessing; future runner must fit scaler on training split only.\n');
fprintf(fid, '- Official 0.5-4 Hz anchor files were checked and not overwritten by this wrapper.\n');
fprintf(fid, '- Upstream post-hoc commit check: `%s`.\n', observed_official_commit);
fprintf(fid, '- Channel inventory: `306` total, `102` mag, `204` grad; labels unique.\n');
fprintf(fid, '- mag102 channel-order SHA256: `%s`.\n', mag_selection.channel_order_sha256);
fprintf(fid, '- Channelwise PSD summary separates mag and grad before aggregation.\n');
fprintf(fid, '- Data provenance hashes record existing local MAT files without modifying them.\n');
fprintf(fid, '- Validation status: passed.\n\n');
fprintf(fid, 'Invalid trials are recorded in `invalid_trials.csv`; each run has one short final trial removed by the official cleanup rule.\n');
fclose(fid);

fprintf('canonical validation passed\n');
end

function [inventoryT, mag_selection] = build_channel_inventory(epochs_neuro, out_dir)
labels = epochs_neuro.label(:);
grad = epochs_neuro.grad;
if ~isfield(grad, 'chantype')
    error('FieldTrip grad.chantype is missing; cannot determine sensor type.');
end
chantype = grad.chantype(:);
if numel(labels) ~= 306 || numel(chantype) ~= 306
    error('Expected 306 labels and chantype entries.');
end
if numel(unique(labels)) ~= numel(labels)
    error('MEG channel labels are not unique.');
end
for trl_idx = 1:numel(epochs_neuro.trial)
    if size(epochs_neuro.trial{trl_idx}, 1) ~= numel(labels)
        error('Trial %d channel count does not match label order.', trl_idx);
    end
end
is_mag = strcmp(chantype, 'megmag');
is_grad = strcmp(chantype, 'megplanar');
if sum(is_mag) ~= 102 || sum(is_grad) ~= 204
    error('Unexpected sensor counts: total=%d mag=%d grad=%d.', numel(labels), sum(is_mag), sum(is_grad));
end
mag_indices = find(is_mag);
mag102_index = strings(numel(labels), 1);
for idx = 1:numel(mag_indices)
    mag102_index(mag_indices(idx)) = string(idx);
end
inventoryT = table((1:numel(labels))', string(labels), string(chantype), is_mag, is_grad, mag102_index, ...
    'VariableNames', {'native_index_1based', 'channel_label', 'sensor_type', 'is_magnetometer', 'is_gradiometer', 'mag102_index_1based'});
writetable(inventoryT, fullfile(out_dir, 'channel_inventory.csv'));

mag_labels = labels(mag_indices);
mag_selection = struct;
mag_selection.source_channel_count = numel(labels);
mag_selection.selected_channel_count = numel(mag_indices);
mag_selection.source_indices_1based = mag_indices(:)';
mag_selection.ordered_channel_labels = mag_labels(:)';
mag_selection.channel_order_sha256 = sha256_text(strjoin(mag_labels, newline));
mag_selection.selection_rule = 'Select FieldTrip channels whose results.epochs_neuro.grad.chantype equals megmag, preserving native FieldTrip order.';
mag_selection.validation = struct('status', 'passed', 'total_channels', numel(labels), 'mag_channels', sum(is_mag), 'grad_channels', sum(is_grad), 'labels_unique', true, 'all_trials_channel_count_matches_label_order', true);
write_json(fullfile(out_dir, 'mag102_selection.json'), mag_selection);
end

function T = build_channelwise_psd(results, expected_labels, inventoryT)
rows = {};
bands = {[0.5, 4], [4, 8], [8, 16]};
band_names = {'0.5-4', '4-8', '8-16'};
fs = results.epochs_neuro.fsample;
sensor_defs = {
    'mag', find(inventoryT.is_magnetometer);
    'grad', find(inventoryT.is_gradiometer)
};
for f_idx = 1:numel(expected_labels)
    label = expected_labels{f_idx};
    trial_idx = results.mapping_epochs(f_idx, 1):results.mapping_epochs(f_idx, 2);
    for s_idx = 1:size(sensor_defs, 1)
        sensor_name = sensor_defs{s_idx, 1};
        chan_idx = sensor_defs{s_idx, 2};
        channel_rel = zeros(numel(chan_idx), numel(bands));
        for c = 1:numel(chan_idx)
            powers = zeros(1, numel(bands));
            for trl = 1:numel(trial_idx)
                x = double(results.epochs_neuro.trial{trial_idx(trl)}(chan_idx(c), :));
                x = x - mean(x);
                [freq, p1] = one_sided_power(x, fs);
                for b = 1:numel(bands)
                    idx = band_mask(freq, bands{b});
                    powers(b) = powers(b) + trapz(freq(idx), p1(idx));
                end
            end
            powers = powers ./ numel(trial_idx);
            channel_rel(c, :) = powers ./ sum(powers);
        end
        for b = 1:numel(bands)
            rows(end+1, :) = {label, sensor_name, band_names{b}, numel(chan_idx), median(channel_rel(:, b)), mean(channel_rel(:, b))}; %#ok<AGROW>
        end
    end
end
T = cell2table(rows, 'VariableNames', {'run_label', 'sensor_type', 'band_hz', 'channel_count', 'median_relative_power', 'mean_relative_power'});
end

function rows = append_psd_rows(rows, subject, label, signal_name, trials, fs)
bands = {[0.5, 4], [4, 8], [8, 16]};
band_names = {'0.5-4', '4-8', '8-16'};
powers = zeros(1, numel(bands));
for trl_idx = 1:numel(trials)
    x = double(trials{trl_idx});
    x = x(:)';
    x = x - mean(x);
    n = numel(x);
    y = fft(x);
    p2 = abs(y / n).^2;
    p1 = p2(1:floor(n/2)+1);
    freq = fs * (0:floor(n/2)) / n;
    for b = 1:numel(bands)
        idx = band_mask(freq, bands{b});
        powers(b) = powers(b) + trapz(freq(idx), p1(idx));
    end
end
powers = powers ./ numel(trials);
denom = sum(powers);
for b = 1:numel(bands)
    rows(end+1, :) = {subject, label, signal_name, band_names{b}, powers(b), powers(b) / denom}; %#ok<AGROW>
end
end

function idx = band_mask(freq, band)
if band(2) == 16
    idx = freq >= band(1) & freq <= band(2);
else
    idx = freq >= band(1) & freq < band(2);
end
end

function [freq, p1] = one_sided_power(x, fs)
n = numel(x);
y = fft(x);
p2 = abs(y / n).^2;
p1 = p2(1:floor(n/2)+1);
freq = fs * (0:floor(n/2)) / n;
end

function record = file_record(path)
info = dir(path);
if isempty(info)
    error('Missing file for provenance: %s', path);
end
record = struct;
record.path = path;
record.sha256 = sha256_file(path);
record.bytes = info.bytes;
end

function commit = get_git_commit(repo_path)
[status, out] = system(sprintf('git -C "%s" rev-parse HEAD', repo_path));
if status ~= 0
    error('Failed to read git commit for %s: %s', repo_path, out);
end
commit = strtrim(out);
end

function value = ternary(condition, if_true, if_false)
if condition
    value = if_true;
else
    value = if_false;
end
end

function hash = sha256_text(text)
md = java.security.MessageDigest.getInstance('SHA-256');
bytes = uint8(text);
md.update(bytes);
hash = lower(reshape(dec2hex(typecast(md.digest(), 'uint8'))', 1, []));
end

function hash = sha256_file(path)
md = java.security.MessageDigest.getInstance('SHA-256');
fid = fopen(path, 'r');
if fid < 0
    error('Cannot open file for hashing: %s', path);
end
cleanup = onCleanup(@() fclose(fid));
while true
    chunk = fread(fid, 1024 * 1024, '*uint8');
    if isempty(chunk)
        break;
    end
    md.update(chunk);
end
hash = lower(reshape(dec2hex(typecast(md.digest(), 'uint8'))', 1, []));
end

function write_json(path, payload)
fid = fopen(path, 'w');
fprintf(fid, '%s', jsonencode(payload, 'PrettyPrint', true));
fclose(fid);
end
