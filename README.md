# HuBMAP- Hacking the Human Vasculature

The goal of this competition is to segment instances of microvascular structures, including capillaries, arterioles, and venules. This competition was hosted by Kaggle, check more details here: 

https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature

Project Organization
------------


    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── models_output      <- Contains the predictions of validation folder per model
    │   ├── processed      <- Processed data (from raw)
    │   └── raw                        <- The original, immutable data dump.
    │
    ├── models             <- All .pt models
    │
    │── notebooks          <- Jupyter notebooks.
    │   ├── EDA.ipynb                           <- Exploratory Data Analysis 
    │   └── CompareModels.ipynb       <- Comparisson between trained models
    │
    ├── requirements.txt   <- Dependencies for running the code
    │
    └── src                <- Source code for use in this project.
         │
         ├── data           
         │     ├── get_mean_std.py     <- Get Mean and STD for own nomralization
         │     ├── pre_process.py        <- Convert all polygons to masks
         │     └── train_test_val.py          Create random train test validation folders 
         │
         └── models         <- Scripts to train models 
              ├── predict_model.py     <- Create all predictions per model
              ├── train_model_1.py     <- Working script 1 for trianing models
              ├── train_model_2.py     <- Working script 2 for trianing models
              └── train_model.py         <- Utils for calculating metrics 



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

------------

## Data Understanding

To gain an understanding of the data, please refer to the notebook titled "Exploratory Data Analysis (EDA)" located in the "notebooks" directory. This notebook provides comprehensive explanations of the raw data, meta-data, data analysis, and the quantity of annotations, among other pertinent information.

## Data Preparation

The data was prepared by converting the polygon format to a mask format, represented as .PNG files. This conversion was performed using a custom function developed in the "pre_process.py" file located in the "src/data" directory. Our conversion method utilized polygons and an optimized flood fill algorithm.

## Train-Test Split

To create a randomized, yet replicable, distribution of the annotated data, including all masks saved as .PNG files under the "data/processed/all_files" directory, we employed the "train_test_val.py" file found in the "src/data" directory.

## Model Training

Having prepared the data, we proceeded with training several models using "train_model_1" and "train_model_2" files from the "src/models" directory. Throughout the training process, we systematically varied parameters such as epochs, batch size, model types, and transformations, among others. The normalization values for the training dataset were obtained using the "get_mean_std.py" file located in the "src/data" directory. All trained models were saved in the "models/" directory, with only the best-performing ones retained for further use.

## Model Predictions

To evaluate the trained models, we utilized the "predict_model.py" file in the "src/models" directory. This script reads all the models contained in the "models/" directory and, for each model, generates a new directory under "data/models_output" to save the predicted masks for the validation dataset. By providing specific parameters for each model in the script, the models were imported with their respective weights, and predictions were made automatically. This streamlined process allowed for the conversion of all trained models efficiently.

## Model Evaluation

For a comprehensive evaluation of the models, considering both visual inspection of the segmentation and various metrics, we created the notebook titled "CompareModels.ipynb" in the "notebook/" directory. This notebook calculates metrics for each model, showcases examples of predicted masks, establishes correlations between the metrics and meta-data, among other analyses. By utilizing this notebook, we could make informed comparisons between models and make decisions based not only on metrics but also on a multi-factor analysis.