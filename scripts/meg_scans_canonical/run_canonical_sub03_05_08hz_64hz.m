function run_canonical_sub03_05_08hz_64hz()
subject = 'sub-03';
dataset_root = 'E:\decode\data\raw\meg_scans_ds006468';
local_derivatives = 'E:\decode\data\derived\meg_scans_canonical_05_08hz_64hz_v1';
official_repo = 'E:\decode\external\upstream\MEG-SCANS';
expected_official_commit = '32bfc690e28e7591b45d96615c59b2d53b6a7165';

fieldtrip_path = 'E:\decode\external\toolboxes\fieldtrip-master\fieldtrip-master';
amtoolbox_path = 'E:\decode\external\toolboxes\amtoolbox-full-1.6.0\amtoolbox-1.6.0';
mtrf_path = 'E:\decode\external\toolboxes\mTRF-Toolbox-master\mTRF-Toolbox-master\mtrf';

official_envelope = fullfile(dataset_root, 'derivatives', 'stimuli', 'sub-others_preprocessed_audiobook_envelopes_decoding.mat');
official_envelope_info_before = dir(official_envelope);

if ~exist(local_derivatives, 'dir')
    mkdir(local_derivatives);
end
if ~exist(fullfile(local_derivatives, subject), 'dir')
    mkdir(fullfile(local_derivatives, subject));
end
ensure_link_or_dir(fullfile(local_derivatives, subject, 'maxfilter'), fullfile(dataset_root, 'derivatives', subject, 'maxfilter'));

addpath(fieldtrip_path);
ft_defaults;
run(fullfile(amtoolbox_path, 'amtstart.m'));
addpath(mtrf_path);
addpath(fullfile(official_repo, 'helper_functions'));
addpath(fullfile(official_repo, 'speech'));
addpath(fullfile(official_repo, 'speech', 'decoding'));

run(fullfile(official_repo, 'speech', 'settings_speech.m'));
observed_official_commit = get_git_commit(official_repo);
if ~strcmp(observed_official_commit, expected_official_commit)
    error('MEG-SCANS upstream commit mismatch. Expected %s, observed %s.', expected_official_commit, observed_official_commit);
end
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

diary_dir = fullfile(local_derivatives, subject, 'speech');
if ~exist(diary_dir, 'dir')
    mkdir(diary_dir);
end
diary_path = fullfile(diary_dir, 'sub-03_canonical_05_08hz_64hz_generation_diary.txt');
diary(diary_path);
fprintf('Canonical MEG-SCANS 0.5-8 Hz / 64 Hz generation started.\n');
fprintf('subject=%s\n', subject);
fprintf('official_repo=%s\n', official_repo);
fprintf('official_repo_commit=%s\n', observed_official_commit);
fprintf('path2bids=%s\n', settings.path2bids);
fprintf('path2derivatives=%s\n', settings.path2derivatives);
fprintf('use_maxfilter=%d audio_latency=%g bpfreq=[%g %g] trialdur=%g fs_neuro=%g fs_down=%g zscore=%d\n', ...
    settings.use_maxfilter, settings.audio_latency, settings.decoding.bpfreq(1), settings.decoding.bpfreq(2), ...
    settings.decoding.trialdur, settings.decoding.fs_neuro, settings.decoding.fs_down, settings.decoding.zscore);

preprocessing_audiobooks(settings);
preprocessing_audiobooks_decoding(subject, settings);

official_envelope_info_after = dir(official_envelope);
if official_envelope_info_before.datenum ~= official_envelope_info_after.datenum || official_envelope_info_before.bytes ~= official_envelope_info_after.bytes
    error('Official 0.5-4 Hz envelope file changed unexpectedly.');
end

fprintf('Canonical MEG-SCANS 0.5-8 Hz / 64 Hz generation finished.\n');
diary off;
end

function commit = get_git_commit(repo_path)
[status, out] = system(sprintf('git -C "%s" rev-parse HEAD', repo_path));
if status ~= 0
    error('Failed to read git commit for %s: %s', repo_path, out);
end
commit = strtrim(out);
end

function ensure_link_or_dir(link_path, target_path)
if exist(link_path, 'dir')
    return;
end
parent = fileparts(link_path);
if ~exist(parent, 'dir')
    mkdir(parent);
end
cmd = sprintf('cmd /c mklink /J "%s" "%s"', link_path, target_path);
[status, out] = system(cmd);
if status ~= 0
    error('Failed to create junction %s -> %s: %s', link_path, target_path, out);
end
end
