function run_official_olsa_preprocessing_sub03()
subject = 'sub-03';
dataset_root = 'E:\decode\data\raw\meg_scans_ds006468';
local_derivatives = 'E:\decode\data\derived\meg_scans_official_replication_v1';
official_repo = 'E:\decode\external\upstream\MEG-SCANS';

fieldtrip_path = 'E:\decode\external\toolboxes\fieldtrip-master\fieldtrip-master';
amtoolbox_path = 'E:\decode\external\toolboxes\amtoolbox-full-1.6.0\amtoolbox-1.6.0';
mtrf_path = 'E:\decode\external\toolboxes\mTRF-Toolbox-master\mTRF-Toolbox-master\mtrf';

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
settings.decoding.fs_neuro = 1000;
settings.decoding.fs_down = 64;
settings.decoding.olsa.prestim = 0;
settings.decoding.olsa.poststim = 0.5;

diary_path = fullfile(local_derivatives, subject, 'speech', 'sub-03_official_olsa_preprocessing_matlab_diary.txt');
diary(diary_path);
fprintf('Official MEG-SCANS OLSA preprocessing wrapper started.\n');
fprintf('subject=%s path2derivatives=%s\n', subject, settings.path2derivatives);
fprintf('use_maxfilter=%d audio_latency=%g bpfreq=[%g %g] fs_neuro=%g fs_down=%g prestim=%g poststim=%g\n', ...
    settings.use_maxfilter, settings.audio_latency, settings.decoding.bpfreq(1), settings.decoding.bpfreq(2), ...
    settings.decoding.fs_neuro, settings.decoding.fs_down, settings.decoding.olsa.prestim, settings.decoding.olsa.poststim);
preprocessing_olsa_decoding(subject, settings);
fprintf('Official MEG-SCANS OLSA preprocessing wrapper finished.\n');
diary off;
end
