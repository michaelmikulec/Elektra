# Dataset Description
The goal of this competition is to detect and classify seizures and other types of harmful brain activity in electroencephalography (EEG) data. Even experts find this to be a challenging task and often disagree about the correct labels.

This is a code competition. Only a few examples from the test set are available for download. When your submission is scored the test folders will be replaced with versions containing the complete test set.

## Files
### train.csv 
Metadata for the train set. The expert annotators reviewed 50 second long EEG samples plus matched spectrograms covering 10 a minute window centered at the same time and labeled the central 10 seconds. Many of these samples overlapped and have been consolidated. train.csv provides the metadata that allows you to extract the original subsets that the raters annotated.
* eeg_id - A unique identifier for the entire EEG recording.
* eeg_sub_id - An ID for the specific 50 second long subsample this row's labels apply to.
* eeg_label_offset_seconds - The time between the beginning of the consolidated EEG and this subsample.
* spectrogram_id - A unique identifier for the entire EEG recording.
* spectrogram_sub_id - An ID for the specific 10 minute subsample this row's labels apply to.
* spectogram_label_offset_seconds - The time between the beginning of the consolidated spectrogram and this subsample.
* label_id - An ID for this set of labels.
* patient_id - An ID for the patient who donated the data.
* expert_consensus - The consensus annotator label. Provided for convenience only.
* [seizure/lpd/gpd/lrda/grda/other]_vote - The count of annotator votes for a given brain activity class. The full names of the activity classes are as follows: lpd: lateralized periodic discharges, gpd: generalized periodic discharges, lrd: lateralized rhythmic delta activity, and grda: generalized rhythmic delta activity . A detailed explanations of these patterns is available here.

### test.csv 
Metadata for the test set. As there are no overlapping samples in the test set, many columns in the train metadata don't apply.
* eeg_id
* spectrogram_id
* patient_id

### sample_submission.csv
* eeg_id
* [seizure/lpd/gpd/lrda/grda/other]_vote - The target columns. Your predictions must be probabilities. Note that the test samples had between 3 and 20 annotators.

### train_eegs/ 
EEG data from one or more overlapping samples. Use the metadata in train.csv to select specific annotated subsets. The column names are the names of the individual electrode locations for EEG leads, with one exception. The EKG column is for an electrocardiogram lead that records data from the heart. All of the EEG data (for both train and test) was collected at a frequency of 200 samples per second.

### test_eegs/ 
Exactly 50 seconds of EEG data.

### train_spectrograms/  
Spectrograms assembled EEG data. Use the metadata in train.csv to select specific annotated subsets. The column names indicate the frequency in hertz and the recording regions of the EEG electrodes. The latter are abbreviated as LL = left lateral; RL = right lateral; LP = left parasagittal; RP = right parasagittal.

### test_spectrograms/ 
Spectrograms assembled using exactly 10 minutes of EEG data.

### example_figures/ 
Larger copies of the example case images used on the overview tab.