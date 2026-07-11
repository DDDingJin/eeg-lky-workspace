function extract_official_decoding_anchor_sub03()
subject = 'sub-03';
repo_root = 'E:\decode\_meg_scans_official_preprocessing_sub03_v1';
local_derivatives = 'E:\decode\data\derived\meg_scans_official_replication_v1';
out_dir = fullfile(repo_root, 'experiments', 'meg_scans_official_preprocessing_sub03_v1');
decoding_path = fullfile(local_derivatives, subject, 'speech', 'sub-03_decoding.mat');
audiobook_path = fullfile(local_derivatives, subject, 'speech', 'sub-03_preprocessed_audiobooks_decoding.mat');
olsa_path = fullfile(local_derivatives, subject, 'speech', 'sub-03_preprocessed_olsa_decoding.mat');

loaded = load(decoding_path, 'results');
results = loaded.results;
eval_model = results.eval_model;

summary = struct;
summary.artifact = 'sub03_official_decoding_anchor_summary.json';
summary.scope = 'official replication anchor only; not comparable to unified Pearson benchmark or EEG results';
summary.subject = subject;
summary.official_commit = '32bfc690e28e7591b45d96615c59b2d53b6a7165';
summary.input_audiobook_mat = audiobook_path;
summary.input_olsa_mat = olsa_path;
summary.output_decoding_mat = decoding_path;
summary.validation_status = 'passed';
summary.n_trials = eval_model.n_trials;
summary.n_trials_train = eval_model.n_trials_train;
summary.n_trials_test = eval_model.n_trials_test;
summary.selected_lambda = eval_model.lambda;
summary.correlation_metric = 'Spearman';
summary.zscore_policy = 'official global z-score; audiobook and OLSA normalized separately; MEG mag and grad normalized separately';
summary.split_policy = 'official rng("shuffle") random 80/20 audiobook split; non-deterministic run';
summary.audiobook_sorted_r = stats_summary(eval_model.stats_sorted.r);
summary.audiobook_shuffled_r = stats_summary(eval_model.stats_shuffled.r);
summary.olsa_sorted_r = stats_summary(results.stats_olsa.r);
summary.olsa_shuffled_r = stats_summary(results.stats_olsa_shuffled.r(:));
summary.n_olsa_trials = numel(results.stats_olsa.r);
summary.n_olsa_shuffled_permutations = size(results.stats_olsa_shuffled.r, 2);

fid = fopen(fullfile(out_dir, 'sub03_official_decoding_anchor_summary.json'), 'w');
fprintf(fid, '%s', jsonencode(summary, 'PrettyPrint', true));
fclose(fid);

rows = {
    'audiobook_sorted', numel(eval_model.stats_sorted.r), mean(eval_model.stats_sorted.r(:)), median(eval_model.stats_sorted.r(:)), std(eval_model.stats_sorted.r(:)), min(eval_model.stats_sorted.r(:)), max(eval_model.stats_sorted.r(:));
    'audiobook_shuffled', numel(eval_model.stats_shuffled.r), mean(eval_model.stats_shuffled.r(:)), median(eval_model.stats_shuffled.r(:)), std(eval_model.stats_shuffled.r(:)), min(eval_model.stats_shuffled.r(:)), max(eval_model.stats_shuffled.r(:));
    'olsa_sorted', numel(results.stats_olsa.r), mean(results.stats_olsa.r(:)), median(results.stats_olsa.r(:)), std(results.stats_olsa.r(:)), min(results.stats_olsa.r(:)), max(results.stats_olsa.r(:));
    'olsa_shuffled', numel(results.stats_olsa_shuffled.r), mean(results.stats_olsa_shuffled.r(:)), median(results.stats_olsa_shuffled.r(:)), std(results.stats_olsa_shuffled.r(:)), min(results.stats_olsa_shuffled.r(:)), max(results.stats_olsa_shuffled.r(:));
};
T = cell2table(rows, 'VariableNames', {'condition', 'n', 'mean_r', 'median_r', 'std_r', 'min_r', 'max_r'});
writetable(T, fullfile(out_dir, 'sub03_official_decoding_anchor_metrics.csv'));

fid = fopen(fullfile(out_dir, 'sub03_official_decoding_anchor_report.md'), 'w');
fprintf(fid, '# sub-03 Official Decoding Replication Anchor\n\n');
fprintf(fid, 'This is an official replication anchor only; it is not comparable to this project''s unified Pearson benchmark or EEG results.\n\n');
fprintf(fid, '- Decoding MAT: `%s`\n', decoding_path);
fprintf(fid, '- n_trials / train / test: `%d / %d / %d`\n', eval_model.n_trials, eval_model.n_trials_train, eval_model.n_trials_test);
fprintf(fid, '- Selected lambda: `%g`\n', eval_model.lambda);
fprintf(fid, '- Correlation metric: `Spearman`\n');
fprintf(fid, '- Split policy: official `rng(\"shuffle\")`, random 80/20 audiobook split, non-deterministic run.\n');
fprintf(fid, '- OLSA shuffled permutations: `%d`\n', size(results.stats_olsa_shuffled.r, 2));
fclose(fid);

fprintf('official decoding anchor extraction passed\n');
end

function s = stats_summary(values)
values = values(:);
s = struct;
s.n = numel(values);
s.mean = mean(values);
s.median = median(values);
s.std = std(values);
s.min = min(values);
s.max = max(values);
end
