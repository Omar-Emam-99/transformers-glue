# Glue Task - Sentiment Analysis

## Description 

* This repo for predicting the sentiment of movie reviews, Most sentiment prediction systems work just by looking at words in isolation, giving positive points for positive words and negative points for negative words and then summing up these points, That way, the order of words is ignored and important information is lost, In constrast, our new deep learning model actually builds up a representation of whole sentences based on the sentence structure. It computes the sentiment based on how words compose the meaning of longer phrases. This way, the model is not as easily fooled as previous models.

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#dataset">Dataset</a></li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
This project aims to build a machine learning pipelines using DVC (Data Version Control) :
1. manage machine learning experiments and pipelines.
2. Tool for data and model versioning.
3. Data access, sharing and collaboration tool.
4. Link between your code and data.
So, this project consists of notebooks and scripts to run pipeline with five stage :
* Stage 1 : Data Loading 
* Stage 2 : Data Preprocessing
* Stage 3 : Train Model
* Stage 4 : Test Model

Model Structure : 
1. Body : 
- Bert Encoder : is pretrained with the two objectives of predicting masked tokens in texts
  and determining if one text passage is likely to follow another. 8 The former task is
  called masked language modeling (MLM) and the latter next sentence prediction (NSP).
2. Classification Head

Training a Text Classifier
- Fine-Tuning Bert-model and Training the hidden states that serve as inputs to the classification model will help us avoid the problem of working with data that may not be well suited for the classifica‐tion task.
 Instead, the initial hidden states adapt during training to decrease the model loss and thus increase its performance.
- I train this model for [5 labels , 3 labels , binary labels] and get higher score on this task .
- so , will show you how to do exprements on this 3 approaches.
    

<!-- dataset -->
### Dataset 
---
The Stanford Sentiment Treebank : movie reviews of five labels ['very negative','negative','nutral','postive','postive'] in PTB Tree format. [<a href='https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf'>paper</a>]
* For difficulty of dataset with 5 labels to get high score, this project solve classification task in three ways :
1. 5 labels.
2. 3 labels ['postive','nutral','negative'].
3. Binary label ['postive','negative].
 
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built-with
---
- Using Datasets liberary to obtain SST dataset.
- Using Transformers liberary to load Model and tokenizer 
- Using Evaluate liberary to evaluating our model.
- Using DVC to create pipelines and making experements and creating plots .

<!-- Getting Started -->
### Getting Started

- Clone the repo and install all requirements and get started .

<!-- Installation -->
#### Installation
---
1. clone the repo 
    ```
    git clone https://github.com/Omar-Emam-99/transformers-glue.git
    ```
2. install all requirments :
    ```
    pip install -r requirements.txt 
    ```
<!-- Usage -->
## Usage 
---
1. Pull data from S3 :
    ```
      dvc pull 
    ```
- Cause i use my own s3 storage from AWS , you need to remote SST data [<a href='https://nlp.stanford.edu/sentiment/index.html'>SST Data</a>] and use `data_utilits.py` from `src/treebank_utils` to convert it to csv formate then remote it on your claud storage like AWS s3 or Google drive :
    ```
    dvc remote add myremote s3://mybucket
    ```
- now we can push entire data cache from the current workspace to the default remote:
    ```
    dvc push 
    ```
- show status or current action :
    ```
    dvc status --cloud
    ```
2. Now we can do experments with specific stage or entire pipeline :
    - show pipeline
    ```
    dvc dag
    ```
    - run entire pipeline
    ```
    dvc repro
    ```
3. dvc.yaml :
  - contain every stage of pipeline that run by dvc 

4. params.yaml
  - you can change parameters or hyperparameters of model and run new experment :
      ```
      dvc exp run
      ```
      - we can change data labels from being 5 class to be 2 or 3 and run exp :
        from `trainer` : `num_labels : 5`
5. Show plots with DVC :
    ```
    dvc plots show 
    ```

<!-- Acknowledgments -->
## Acknowledgments

- Glue Tasks <a href="https://nlp.stanford.edu/sentiment/index.html">The Stanford Sentiment Treebank</a>
- NLP with Transformers Book
- DVC from <a href="https://iterative.ai/">iterative.ai</a>
