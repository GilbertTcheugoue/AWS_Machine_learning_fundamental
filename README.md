# AWS_Machine_learning_fundamental
# Predict Bike Sharing Demand with AutoGluon project

## Introduction to AWS Machine Learning Final Project

## Overview
In this project, students will apply the knowledge and methods they learned in the Introduction to Machine Learning course to compete in a Kaggle competition using the AutoGluon library.

Students will create a Kaggle account if they do not already have one, download the Bike Sharing Demand dataset, and train a model using AutoGluon. They will then submit their initial results for a ranking.

After they complete the first workflow, they will iterate on the process by trying to improve their score. This will be accomplished by adding more features to the dataset and tuning some of the hyperparameters available with AutoGluon.

Finally they will submit all their work and write a report detailing which methods provided the best score improvement and why. A template of the report can be found [here](report-template.md).

To meet specifications, the project will require at least these files:
* Jupyter notebook with code run to completion
* HTML export of the jupyter notebbook
* Markdown or PDF file of the report

Images or additional files needed to make your notebook or report complete can be also added.

## Getting Started
* Clone this template repository `git clone git@github.com:udacity/nd009t-c1-intro-to-ml-project-starter.git` into AWS Sagemaker Studio (or local development).

<img src="img/sagemaker-studio-git1.png" alt="sagemaker-studio-git1.png" width="500"/>
<img src="img/sagemaker-studio-git2.png" alt="sagemaker-studio-git2.png" width="500"/>

* Proceed with the project within the [jupyter notebook](project-template.ipynb).
* Visit the [Kaggle Bike Sharing Demand Competition](https://www.kaggle.com/c/bike-sharing-demand) page. There you will see the overall details about the competition including overview, data, code, discussion, leaderboard, and rules. You will primarily be focused on the data and ranking sections.

### Dependencies

```
Python 3.7
MXNet 1.8
Pandas >= 1.2.4
AutoGluon 0.2.0 
```

### Installation
For this project, it is highly recommended to use Sagemaker Studio from the course provided AWS workspace. This will simplify much of the installation needed to get started.

For local development, you will need to setup a jupyter lab instance.
* Follow the [jupyter install](https://jupyter.org/install.html) link for best practices to install and start a jupyter lab instance.
* If you have a python virtual environment already installed you can just `pip` install it.
```
pip install jupyterlab
```
* There are also docker containers containing jupyter lab from [Jupyter Docker Stacks](https://jupyter-docker-stacks.readthedocs.io/en/latest/index.html).

## Project Instructions

1. Create an account with Kaggle.
2. Download the Kaggle dataset using the kaggle python library.
3. Train a model using AutoGluon’s Tabular Prediction and submit predictions to Kaggle for ranking.
4. Use Pandas to do some exploratory analysis and create a new feature, saving new versions of the train and test dataset.
5. Rerun the model and submit the new predictions for ranking.
6. Tune at least 3 different hyperparameters from AutoGluon and resubmit predictions to rank higher on Kaggle.
7. Write up a report on how improvements (or not) were made by either creating additional features or tuning hyperparameters, and why you think one or the other is the best approach to invest more time in.

## License
[License](LICENSE.txt)


# Project Overview

# Building a Handwritten Digit Classifier
At the end of the course, you'll build a handwritten digit classifier by:

Setting up a Jupyter notebook and importing the necessary libraries
Loading and preprocessing your image data
Designing and building your neural network in PyTorch
Choosing an optimizer and loss function
Training your model
Evaluating your model
Optimizing your hyperparameters or training process
This will ultimately allow you to train a model that recognizes handwritten digits like those in the image below, a key step in optical character recognition.



# Project Instructions

# Step 1
Load the dataset from torchvision.datasets
Use transforms or other PyTorch methods to convert the data to tensors, normalize, and flatten the data.
Create a DataLoader for your dataset
# Step 2
Visualize the dataset using the provided function and either:
Your training data loader and inverting any normalization and flattening
A second DataLoader without any normalization or flattening
Explore the size and shape of the data to get a sense of what your inputs look like naturally and after transformation. Provide a brief justification of any necessary preprocessing steps or why no preprocessing is needed.
# Step 3
Using PyTorch, build a neural network to predict the class of each given input image
Create an optimizer to update your network’s weights
Use the training DataLoader to train your neural network
# Step 4
Evaluate your neural network’s accuracy on the test set.
Tune your model hyperparameters and network architecture to improve your test set accuracy, achieving at least 90% accuracy on the test set.

# Step 5
Use torch.save to save your trained model.
