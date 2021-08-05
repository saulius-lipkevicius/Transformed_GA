

<!-- PROJECT LOGO -->
<br />
<p align="center">
    <img src="./../images/logo.png" alt="Logo" width="160" height="160">
  </a>

  <h3 align="center">Transformer BERT </h3>

  <p align="center">
    <a href="https://github.com/"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/">View Demo</a>
    ·
    <a href="https://github.com/">Report Bug</a>
    ·
    <a href="https://github.com/">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-bert">About The BERT</a>
      <ul>
        <li><a href="#model-advantages">Model advantages</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#colab-setup">Colab Setup</a></li>
        <li><a href="#preprocess-and-data-locate">Preprocess and Data Locate</a></li>
      </ul>
    </li>
    <li><a href="#tokenization-&-input-formatting">Tokenization & input Formatting</a>
    </li>
    <ul>
        <li><a href="#special-tokens">Special Tokens</a></li>
        <li><a href="#aptamer-length">Aptamer Length</a></li>
      </ul></li>
      <li><a href="train">Train</a>
      <ul>
        <li><a href="#model">Model</a></li>
        <li><a href="#hyperparameters">Hyperparameters</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a>
    </li>
    <li><a href="#model-performance">Model Performance</a>
    </li>
    <li><a href="#further-improvements">Further Improvements</a>
    </li>
    <li><a href="#appendix">Appendix</a>
    <ul>
        <li><a href="#saving&loading-fine-tuned-model">Saving & Loading fine-tuned Model</a></li>
        <li><a href="#optimizer-&-learning-rate-scheduler">Optimizer & Learning Rate Scheduler</a></li>
        <li><a href="#weight-decay">Weight Decay</a></li>
      </ul>
    <li><a href="#acknowledgements">Acknowledgements</a>
  </ol>
</details>


Model Performance
Further improvements
2 Appendix
<!-- ABOUT THE PROJECT -->
## About The BERT
----
2018 was a breakthrough year for NLP, multiple new models like OpenAI's Open-GTP, Google's BERT allowed researchers to fine-tune existing models to produce state-of-art performance with minimal effort.

BERT (Bidirectional Encoder Representations from Transformers) was released in late 2018, initialy model was employed to Natural Language Processing (NLP). It is used to extract high quality language features, in our case aptamer sequences. (prideti apie galimybes klasifikuoti, sentiment analysis, ir t.t.)
In this project, we will use BERT to train a sequence classifier, more specifically, we will take a dataset with sequences comparison (entropy score) to fine-tune a model to be able to compare random sequences entropy. (Prideti kodel sekos gali buti laikomos kalbine israiska)

It differs from base BERT model by one untrained layer of neurons on the end and then we train. Many other IGEM teams have tried to apply longer established deep learning methods, for instance CNN, LSTM and so on. Lets consider why transformers have an edge over former models.

### Model advantages
* **Quick Development**

  - To fine-tune BERT model on specific sequential data task we don't need to train original BERT model or any other from scratch (GPU expensive and time) because bottom layers are already trained only top classifying layers. (patikslinti)

* **Less Data**
  - Top deep learning models have millions of parameters to train, therefore model from scratch requires immense size datasets, a lot of time, and hands to create dataset.
  
* **Better Results**
    - It was shown that simple fine-tuning, by adding one layer on the top of BERT, can archieve state-of-art results with minimal task-specific adjustments.


## Getting Started 
----
### Colab Setup
Google Colabatory offers free GPUs which is perfect to train large neural networks like BERTs. To add GPU select on menu:

`Edit -->  Notebook Settings --> Hardware accelerator --> (GPU)`

This Framework has some drawbacks if model training takes long time, hence we used Google Colabatory Pro to speed up the process.


### Preprocess and Locate Data
Model input must follow standard norms - (sequence1, sequence2, label). In case you have sequences from **M.A.W.S** you have to run it through *Python* script *pairing.py*

* Pairing aptamers to fit it into *BERT*
  ```sh
  python pairing.py -d yourCSVsheetWithAptamers
  ```

  PRIDETI SNAPSHOTA dataframe


### Dependencies

To use a pre-trained transformer models *HuggingFace* provides API to quickly download and use those on a give dataset. API contains thousands of pretrained models to perform many tasks, however in our case we employed it for classification. (https://pypi.org/project/transformers) it gives *pytorch* interface for working with *BERT*. (kodel pasirinkome pytorch). It includes variety of models, also includes and pre-build modifacionts of these models suited for variety of tasks, for instance, we will use `BertForSequenceClassification`.

* Install transformers
  ```sh
  !pip install transformers
  ```

* To download and use model
  ```sh
  from transformers import AutoTokenizer, AutoModel

  tokenizer = AutoTokenizer.from_pretrained(bert_model) 
  bert_model = AutoModel.from_pretrained(bert_model)
  ```

  In case you have specific data, you can train *tokenizer* on your own. (https://huggingface.co/quicktour/transformers). More considerations can be found in [Further_improvements]


## Tokenization & input Formatting
----
Formating required is:
  - *Special* tokens at the beginning and ending of each sentence.
  - Padding & truncation to a single constant lenght. (papildyti)
  - Differ *real* tokens from *padding* tokens with attention mask.

### Special Tokens

`[CLS]` - For classification tasks, this token is appended in the beginning of first sentence. The significance of this token appears after all embeddings and produce classifier value - prediction.

`[SEP]` - Appears in the ending of *every* sentence and is given to seperate sentences to help model determine something.

`[PAD]` - Is used to balance every input sequence lenghts.

ITERPTI ARCHITEKTUROS PNG
<p align="center">
    <img src="./../images/logo.png" alt="Logo" width="160" height="160">
  </a>

* How to looks in our case
  ```sh
  print("Original input: 'ACGTTGAACG', 'CGTTTCGAAT' ")
  print('Tokenized: ', tokenizer("ACGTTGAACG", "CGTTTCGAAT")['input_ids'])
  print('Seperating sequences: ', tokenizer("ACGTTGAACG", "CGTTTCGAAT")['token_type_ids'])
  ```

   ```sh
  Original input: 'ACGTTGAACG', 'CGTTTCGAAT' 
  Tokenized:  [2, 21, 15123, 38, 38, 1136, 1738, 263, 3, 13, 15123, 38, 38, 6668, 1136, 721, 3]
  Seperating sequences:  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  ```

### Aptamer Length

In case dataset consists varying aptamer lengths we have to consider two *BERT* constraints:

* All aptamer pairs must be padded or truncated to a same, fixed length.

* The maximum lump length can't exceed 512 tokens.

Check *Class CustomerDataset* how it should be implemented. However, the last thing to consider is training time, training time is approximaly linearly dependent on *max_len*. (linkas)
   
## Train
----
Following good practices data was divided up in *train*, *test*, *validation* groups with *80%*, *10%*, *10%* respectively, refer to *pairing.py*.
Next, an iterator for our dataset using *torch DataLoauder* class is created, which helps to save memory compared to simply looping data and whole data loaded to memory.


* train, validation datasets
  ```sh
  train_loader = DataLoader(train_set, batch_size=bs, num_workers=1)
  val_loader = DataLoader(val_set, batch_size=bs, num_workers=1)
  ```
  Now, if we have a worker process, we can make use of the fact that our machine has multiple cores. This means that the next batch can already be loaded and ready to go by the time the main process is ready for another batch. This is where the speed up comes from. The batches are loaded using additional worker processes and are queued up in memory. Optimal number of workers is equal 1.[https://deeplizard.com/learn/video/kWVgvsejXsE]


### Model
[https://huggingface.com/transformers] lists all possible *BERT* alternatives/modifications to choose from.

### Hyperparameters
Authors suggest to use *learning rate = x* (saltinis)
Also, to optimize training time, we suggest to test *batch_size = y*, however it is related to learning rate linearly. (saltinis)


<!-- USAGE EXAMPLES -->
## Usage
----
We have created a function to simplify usage of model:
* How to use model
  ```sh
  test_prediction(net=model, device=device, dataloader=test_loader, with_labels=True, result_file=path_to_output_file)
  ```
**padaryti hugging face modeliu**

_For more examples, please refer to the [Documentation](https://example.com)_ --->GAL CIA IKISTI HUGGINGFACE modeli parsisiuntima

## Model performance
1. iverciai ir grafikeliai training/val loss
2. test data ivertinimas (gal paieskoti kitu iverciu ) + confuson matrica
----

## Further improvements
  - It is possible to create a tokenizer that learns to distinquish the most important parts of sequence, especially if some base seeks are of interest, for instance hairpins. Fit tokenizer might improve transformers efficiency.
  - In case you want to push model even further and employ *large BERT* modifications/alternatives, you should expand a dataset to help model train that massive number of parameters.

## Appendix
### A1.Saving & Loading Fine-tuned Model
### A2.Optimizer & Learning Rate Scheduler
### A3.Weight Decay
1. Kaip pasirinkti optimezeri ir learning rate
2. batch size su learning rate santykis
3. kas yra svorio kritimas
4. ar laudinti modeli, t.y. sioje skiltyje???

## Acknowledgements
----
 
1. [HuggingFace](https://huggingface.co) :hugs: documentation to pick the most suitable model for your task.
2. i paperius apie svorius learning rate ir etc 

