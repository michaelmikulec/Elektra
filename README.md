# Elektra: Harmful Brain Activity Classifier
## Objective
* Classify input EEG datasets into one of 6 categories:
  1. Seizure
  2. LPD 
  3. GPD
  4. LRDA
  5. GRDA
  6. Other

## Features
* Contains 3 learning models:
  1. Random Forest machine learning model
  2. Transformer deep learning model
  3. CNN deep learning model

* Models are pre-trained on EEG and Spectrogram Data and are ready to be used to make inferences on new data

## UI Module

### Requirements
  * Input: accepts parquet or csv file with EEG signal data or spectrogram data
  * Output: prints classification of input file to screen

### Buttons
  * upload file
  * run machine learning inferencer
  * run deep learning tranformer inferencer
  * run deep learning CNN inferencer

### Display Elements
  * display accuracy of model
  * display inferred classification of input file
  * display confidence of inference

## Machine Learning Module
### Random Forest Model

## Deep Learning Module
### Transformer Model
### CNN Model


