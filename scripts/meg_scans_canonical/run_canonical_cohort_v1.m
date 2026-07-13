function run_canonical_cohort_v1()
% Generate sensor-level canonical paired data only; no model training.
subjects = {'sub-01','sub-02','sub-03','sub-04','sub-05','sub-06','sub-07','sub-08','sub-09','sub-10','sub-11','sub-12','sub-13','sub-14','sub-15','sub-16','sub-17','sub-18','sub-19','sub-20','sub-21','sub-22','sub-23','sub-24'};
dataset_root = 'E:\decode\data\raw\meg_scans_ds006468';
local_derivatives = 'E:\decode\data\derived\meg_scans_cohort_canonical_05_08hz_64hz_v1';
official_repo = 'E:\decode\external\upstream\MEG-SCANS';
expected_official_commit = '32bfc690e28e7591b45d96615c59b2d53b6a7165';
fieldtrip_path = 'E:\decode\external\toolboxes\fieldtrip-master\fieldtrip-master';
amtoolbox_path = 'E:\decode\external\toolboxes\amtoolbox-full-1.6.0\amtoolbox-1.6.0';
mtrf_path = 'E:\decode\external\toolboxes\mTRF-Toolbox-master\mTRF-Toolbox-master\mtrf';

if ~exist(local_derivatives, 'dir'), mkdir(local_derivatives); end
addpath(fieldtrip_path); ft_defaults;
run(fullfile(amtoolbox_path, 'amtstart.m'));
addpath(mtrf_path); addpath(fullfile(official_repo, 'helper_functions'));
addpath(fullfile(official_repo, 'speech')); addpath(fullfile(official_repo, 'speech', 'decoding'));
observed = get_git_commit(official_repo);
if ~strcmp(observed, expected_official_commit)
    error('MEG-SCANS upstream commit mismatch: %s', observed);
end
run(fullfile(official_repo, 'speech', 'settings_speech.m'));
settings.path2bids = dataset_root;
settings.path2derivatives = local_derivatives;
settings.path2fieldtrip = fieldtrip_path;
settings.path2amtoolbox = amtoolbox_path;
settings.path2mtrftoolbox = mtrf_path;
settings.use_maxfilter = true;
settings.apply_latency_correction = true;
settings.audio_latency = 3/1000;
settings.fs_audio = 44100;
settings.decoding.filtertype = 'firws';
settings.decoding.bpfreq = [0.5, 8];
settings.decoding.trialdur = 120;
settings.decoding.fs_neuro = 1000;
settings.decoding.fs_down = 64;
settings.decoding.zscore = false;

% Envelope generation is shared and written only below local_derivatives.
preprocessing_audiobooks(settings);
for idx = 1:numel(subjects)
    subject = subjects{idx};
    if ~has_sensor_inputs(subject, dataset_root)
        fprintf('SKIP %s: missing raw/maxfiltered/events input\n', subject);
        continue;
    end
    subject_dir = fullfile(local_derivatives, subject);
    if ~exist(subject_dir, 'dir'), mkdir(subject_dir); end
    ensure_link_or_dir(fullfile(subject_dir, 'maxfilter'), fullfile(dataset_root, 'derivatives', subject, 'maxfilter'));
    diary_dir = fullfile(subject_dir, 'speech');
    if ~exist(diary_dir, 'dir'), mkdir(diary_dir); end
    diary(fullfile(diary_dir, [subject '_canonical_05_08hz_64hz_generation_diary.txt']));
    fprintf('Canonical generation started subject=%s official_commit=%s\n', subject, observed);
    try
        preprocessing_audiobooks_decoding(subject, settings);
        fprintf('Canonical generation passed subject=%s\n', subject);
    catch err
        fprintf('Canonical generation FAILED subject=%s reason=%s\n', subject, err.message);
    end
    diary off;
end
end

function tf = has_sensor_inputs(subject, root)
tasks = {'audiobook1','audiobook2'};
tf = true;
for t = 1:numel(tasks)
    for r = 1:2
        stem = sprintf('%s_task-%s_run-%02d', subject, tasks{t}, r);
        tf = tf && exist(fullfile(root, subject, 'meg', [stem '_meg.fif']), 'file') == 2;
        tf = tf && exist(fullfile(root, 'derivatives', subject, 'maxfilter', [stem '_proc-tsss-mc_meg.fif']), 'file') == 2;
        tf = tf && exist(fullfile(root, subject, 'meg', [stem '_events.tsv']), 'file') == 2;
    end
end
end

function commit = get_git_commit(repo_path)
[status, out] = system(sprintf('git -C "%s" rev-parse HEAD', repo_path));
if status ~= 0, error('Failed to read upstream commit: %s', out); end
commit = strtrim(out);
end

function ensure_link_or_dir(link_path, target_path)
if exist(link_path, 'dir'), return; end
parent = fileparts(link_path);
if ~exist(parent, 'dir'), mkdir(parent); end
cmd = sprintf('cmd /c mklink /J "%s" "%s"', link_path, target_path);
[status, out] = system(cmd);
if status ~= 0, error('Failed to create junction %s -> %s: %s', link_path, target_path, out); end
end
