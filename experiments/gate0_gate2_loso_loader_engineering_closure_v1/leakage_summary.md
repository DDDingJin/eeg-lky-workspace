# Leakage Summary

All jobs in this engineering closure keep pure LOSO semantics: the heldout subject is excluded from train, val, normalization fitting, and checkpoint selection, and used only for test.

| model | heldout_subject | excluded_from_train | excluded_from_val | excluded_from_selection | test_subjects |
| --- | --- | --- | --- | --- | --- |
| eegnet | P00 | True | True | True | P00 |
| eegnet | P01 | True | True | True | P01 |
| ridge | P00 | True | True | True | P00 |
| ridge | P01 | True | True | True | P01 |
