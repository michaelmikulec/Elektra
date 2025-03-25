import labeler
import eeg_dataset_class
import eeg_transformer
import transformer_trainer
import spec_dataset_class
import spec_cnn
import cnn_trainer
import inferencer
import multiprocessing as mp

labels = ["Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"]
label_index = {
    "Seizure": 0,
    "LRDA": 1, 
    "GRDA": 2, 
    "LPD": 3, 
    "GPD": 4, 
    "Other": 5
}

mp.set_start_method("spawn", force=True)