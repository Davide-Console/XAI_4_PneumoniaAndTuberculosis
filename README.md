# AAIB Assignment 2022: X-Ray images classifier for PNEUMONIA and TUBERCULOSIS

## Setup 

* Open a command prompt and execute:
```console
https://github.com/RaffaeleBerzoini/AAIB_Assignment2022.git
cd AAIB_Assignment2022
```
Download the [dataset](https://drive.google.com/drive/folders/1KS4tFoB1SZU6HLhm0pfMPmtNOOrFB1Az) and place the **train_set.zip** file in the project folder

The working directory should look similar to:

```text
AAIB_Assignment2022   # your WRK_DIR
.
├── AE_model
    ├── 0.0103f_model.h5
├── explainedModels
├── ChartsAndPlots
├── .py scripts
├── train_set.zip
└── .py scripts
```

The project has been tested with tensorflow 2.4.1

* In the command prompt execute:
```console
python dataset_preparation.py
```

## Dataset exploration

To visualize some dataset information:
```console
python data_exploration.py
```


## Denoising techniques

To visualize some denoising techniques examples:
```console
python denoising_comparison.py
```


## Architecture

To visualize the DL architecture used in training:
```console
python architectures.py
```


## Training

To train the best DL model:

  ```console
python train.py
  ```

During training, each time validation results improve, a float model is saved in:
`float_model/{val_accuracy:.4f}-{accuracy:.4f}-f_model.h5`

A csv with train and validation accuracy and loss trends is saved in `./train_log.csv`

To perform cross validation on a sklearn model (SVC):

  ```console
python sklearn_model_train.py
  ```

## Testing

To evaluate results:
  ```console
python evaluate.py
python evaluate_SVM.py
  ```

## XAI

You can change some parameters inside the scripts to personalize the output and the model used

-Lime:
  ```console
python XAI_Lime.py
  ```

-GradCam:
  ```console
python XAI_GradCam.py
  ```

-Occlusion:
  ```console
python XAI_Occlusion.py
  ```

-Inverted Occlusion:
  ```console
python XAI_InvertedOcclusion.py
  ```