# Argument Mining Module Project

Module project for module grade


Installation and functionality description
=======================


-------

1. Intro

This directory contains two main scripts as Python files written to build a pipeline to train and evaluate two types of models.

The pipeline has four main parts: preprocessing, feature extraction, model training and evaluation of the results. 


The directory contains:

* `classify_rfc.py`
* `classify_transformers.py`
* `preprocess.py`
* `features.py`
* `evaluate.py`
* `essays_semantic_types.tsv`
* `README.md`(this file)
* `.gitignore`
* `requirements.txt`
* ArgMinBericht.pdf
* corpus(directory with article files)


-------

2. Installation

1) Clone the repository.

2) Using your terminal navigate through your computer to find the directory were you cloned the repository. Then from Terminal (look for 'Terminal' on Spotlight), or CMD for Windows,  set your working directory to that of your folder (for example: cd Desktop/ArgMin_Modulprojekt).

3) Required packages:

The requirements.txt file should install all the dependencies when you run the script from your IDE.

If for some reason that does not work and you don't have pip installed follow the installing instructions here: https://pip.pypa.io/en/stable/installation/

Install the required packages specified on the .txt file by typing on your terminal:

```
pip install required_package
```


4You should be able to run the script now. Check first how you can run python on your computer (it can be 'python' or 'python3'). The program will generate several files, including the used models, data splits, accuracies of all models evaluated and their classification reports.

-------
3. Use

1) You need to run first the `classify_transformers.py` file. There we reformat the corpus and generate the files and data we will need for the rest of the process. The training, dev and test sets are also created at this stage and will get saved for later access.

2) After we generate the needed files the transformer model will be trained and evaluated. The respective report will also get saved in a file.
3) When the previous steps are finished you can run the `classify_rfc.py` file. This file will recover previously generated datasets and use them to train a RandomForestClassifier model, that will also be evaluated and will generate and save a report.
-------
4. Contact information

If you have any questions or problems during they installation process, feel free to email sandra.sanchez.paez@uni-potsdam.de