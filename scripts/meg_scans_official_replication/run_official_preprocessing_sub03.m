function run_official_preprocessing_sub03()
% Local-only wrapper for official MEG-SCANS audiobook decoding preprocessing.

subject = 'sub-03';
dataset_root = 'E:\decode\data\raw\meg_scans_ds006468';
local_derivatives = 'E:\decode\data\derived\meg_scans_official_replication_v1';
official_repo = 'E:\decode\external\upstream\MEG-SCANS';

fieldtrip_path = 'E:\decode\external\toolboxes\fieldtrip-master\fieldtrip-master';
amtoolbox_path = 'E:\decode\external\toolboxes\amtoolbox-full-1.6.0\amtoolbox-1.6.0';
mtrf_path = 'E:\decode\external\toolboxes\mTRF-Toolbox-master\mTRF-Toolbox-master\mtrf';

if ~exist(local_derivatives, 'dir')
    mkdir(local_derivatives);
end
if ~exist(fullfile(local_derivatives, subject), 'dir')
    mkdir(fullfile(local_derivatives, subject));
end
ensure_link_or_dir(fullfile(local_derivatives, subject, 'maxfilter'), fullfile(dataset_root, 'derivatives', subject, 'maxfilter'));
ensure_link_or_dir(fullfile(local_derivatives, 'stimuli'), fullfile(dataset_root, 'derivatives', 'stimuli'));

addpath(fieldtrip_path);
ft_defaults;
run(fullfile(amtoolbox_path, 'amtstart.m'));
addpath(mtrf_path);
addpath(fullfile(official_repo, 'helper_functions'));
addpath(fullfile(official_repo, 'speech'));
addpath(fullfile(official_repo, 'speech', 'decoding'));

run(fullfile(official_repo, 'speech', 'settings_speech.m'));
settings.path2bids = dataset_root;
settings.path2derivatives = local_derivatives;
settings.path2fieldtrip = fieldtrip_path;
settings.path2amtoolbox = amtoolbox_path;
settings.path2mtrftoolbox = mtrf_path;
settings.use_maxfilter = true;
settings.apply_latency_correction = true;
settings.audio_latency = 3/1000;
settings.decoding.filtertype = 'firws';
settings.decoding.bpfreq = [0.5, 4];
settings.decoding.trialdur = 120;
settings.decoding.fs_neuro = 1000;
settings.decoding.fs_down = 64;

diary_path = fullfile(local_derivatives, subject, 'speech', 'sub-03_official_preprocessing_matlab_diary.txt');
if ~exist(fileparts(diary_path), 'dir')
    mkdir(fileparts(diary_path));
end
diary(diary_path);
fprintf('Official MEG-SCANS preprocessing wrapper started.\n');
fprintf('subject=%s\n', subject);
fprintf('path2bids=%s\n', settings.path2bids);
fprintf('path2derivatives=%s\n', settings.path2derivatives);
fprintf('use_maxfilter=%d audio_latency=%g bpfreq=[%g %g] trialdur=%g fs_neuro=%g fs_down=%g\n', ...
    settings.use_maxfilter, settings.audio_latency, settings.decoding.bpfreq(1), settings.decoding.bpfreq(2), ...
    settings.decoding.trialdur, settings.decoding.fs_neuro, settings.decoding.fs_down);

preprocessing_audiobooks_decoding(subject, settings);
fprintf('Official MEG-SCANS preprocessing wrapper finished.\n');
diary off;
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
