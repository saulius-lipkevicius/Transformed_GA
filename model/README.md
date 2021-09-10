

<!-- PROJECT LOGO -->
<br />
<p align="center">
    <img src="./../images/logo.png" alt="Logo" width="160" height="160">
  </a>

  <h3 align="center">Fine-Tuning Transformer alBERT </h3>

  <p align="center">
    <a href="https://github.com/"><strong>Explore the wiki »</strong></a>
    
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-bert">About BERT</a>
      <ul>
        <li><a href="#model-advantages">Model advantages</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#colab-setup">Colab Setup</a></li>
        <li><a href="#preprocessing-and-data-locating">Preprocessing and Data Locating</a></li>
      </ul>
    </li>
    <li><a href="#tokenization-&-input-formatting">Tokenization & input Formatting</a>
    </li>
    <ul>
        <li><a href="#special-tokens">Special Tokens</a></li>
        <li><a href="#aptamer-length">Aptamer Length</a></li>
      </ul></li>
      <li><a href="training">Training</a>
      <ul>
        <li><a href="#hyperparameters">Hyperparameters</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a>
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


<!-- ABOUT THE PROJECT -->
## About BERT
----
2018 was a breakthrough year for NLP, multiple new models like OpenAI's Open-GTP, Google's BERT allowed researchers to fine-tune existing models to produce state-of-art performance with minimal effort. Almost instantly LSTM was largely replaced by BERT (Bidirectional Encoder Representations from Transformers). BERT was released in the late 2018, initially model was employed to Natural Language Processing (NLP), nowadays it is applied in a variety of fields, for example, machine translation, document generation, biological sequence analysis, and video understanding. 

In this project, we will be using a BERT modification - a Lite BERT (alBERT) that incorporated parameter-reduction techniques to avoid memory limitations of available software, hence it has multiple times less parameters to learn and trains 1.7x faster. Multiple iGEM teams have tried to apply well-established deep learning methods like CNN, LSTM to predict some features of biological sequences, however those architectures have gradient flaws that especially reveal itself in long sequences. 

### Model advantages
---
* **Quick Development**
  - Compared to LSTM, BERT has a property of transfer learning, which means that you do not have to train the lower layers of model from scratch. You only need to apply the *head-layer* that suits your task, to get state-of-art results. 

* **Less Data**
  - Best performing deep learning models have millions of parameters to train, therefore the model from scratch requires datasets of immense size, which takes a lot of time and hands to create.
  
* **Better Results**
    - It was shown that a simple fine-tuning, by adding one layer on the top of BERT, can archieve state-of-art results with minimal task-specific adjustments, and it does not suffer from vanishing/exploding gradient (RNN illness). As a consequence, BERT can handle long sequences.

* **Completely Exhaust GPU resources**
    - RNN, LSTM were hardly parallelizable because of recurrent-like architecture. To avoid this issue BERT employed the new [attention link] methodology that allows BERT to fully parallelize computations.

## Getting Started 
----
### Colab Setup
Google Colaboratory offers free GPUs which makes it a perfect platform to train large neural networks like alBERTs. To add GPU select on menu:

`Edit -->  Notebook Settings --> Hardware accelerator --> (GPU)`

This Framework has some time and resourses drawbacks if training dataset is huge or *large* alBERT architecture is chosen, hence we used Google Colaboratory Pro to speed up the process.


### Preprocessing and Data Locating
Model input must follow standard norms - **(Sequence1, Sequence2, Label)**. In case you have a list of sequences from **M.A.W.S** you have to run it through *Python* script *pairing.py* to generate labelled dataframe for training.

* Pairing aptamers to fit it into *alBERT*
  ```sh
  python pairing.py -d aptamerListCSV
  ```
* Output format
<p align="center">
    <img src="./../images/dataframe.png" alt="dataframe" width="360" height="160">
</a>


### Dependencies

To use a pre-trained transformer *HuggingFace* :hug: provides API to quickly download and use those on a given dataset. API contains thousands of pretrained models to perform many tasks including all *BERT* modifications, however in our case we employed *alBERT* for classification. More information on *pytorch interface*(https://pypi.org/project/transformers)

* Install transformers
  ```sh
  !pip install transformers
  ```

* Download and use model
  ```sh
  from transformers import AutoTokenizer, AutoModel

  tokenizer = AutoTokenizer.from_pretrained(bert_model) 
  bert_model = AutoModel.from_pretrained(bert_model)
  ```

## Tokenization & input Formatting
----
Required formating:
  - *Special* tokens at the beginning and end of each sentence.
  - Padding & truncation to a single constant length.
  - Differ *real* tokens from *padding* tokens with attention mask.

### Special Tokens

`[CLS]` - a token for classification tasks, this token is appended at the beginning of first sentence. The significance of this token appears after all embeddings and produce classifier value - prediction.

`[SEP]` - a token that appears in the ending of *every* sentence and is given to separate sentences to help model determine something.

`[PAD]` - a token that is used to balance every input sequence lenghts.

ITERPTI ARCHITEKTUROS PNG kad butu suprantamiau
<p align="center">
    <img src="./../images/logos.png" alt="Logo" width="160" height="160">
  </a>

* How it looks in our case
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

In case, dataset consists of varying length aptamers we have to consider two *alBERT* constraints:

* Every aptamer pair must be padded or truncated to the same, fixed length.

* The maximum lump length cannot exceed 512 tokens.

 However, keep *max_len* as small as possible, since the training time is approximately linearly dependent on this parameter. [linkas]


## Training
----
Following good practice, data was divided up in *train*, *test*, *validation* groups with *80%*, *10%*, *10%* percentage of data respectively, refer to *pairing.py*.
Next, an *iterator* for our dataset using *torch DataLoauder* class is created, which helps to save memory compared to simply data looping which stucks whole loaded data to memory.

* train, validation datasets
  ```sh
  train_loader = DataLoader(train_set, batch_size=bs, num_workers=1)
  val_loader = DataLoader(val_set, batch_size=bs, num_workers=1)
  ```
  Machine with GPU has multiple cores, which means that the next batch can already be loaded and ready to go by the time the main process is completed and prepared for another batch. This is where the *number_of_workers* comes and speeds up: batches are loaded by workers and queued up in memory.  The optimal number of workers is equal 1.[https://deeplizard.com/learn/video/kWVgvsejXsE]

Check Appendix to check how other hyperparameters can be optimized.

<!-- USAGE EXAMPLES -->
## Usage
----
We have created a function to simplify usage of model:
* How to predict employing model
  ```sh
  test_prediction(net=model, device=device, dataloader=test_loader, with_labels=True, result_file=path_to_output_file)
  ```
**IKELTI I HUGGINGFACE platforma, jeigu tinka musu modelis**

Please refer to the [Model usage helicopter overview](https://huggingface.co/transformers/v1.0.0/model_doc/overview.html).

## Model optimization with ONNX
Transformers and transformers-like achitectures have taken over many sequence related field with de-facto state-of-art performance, however it comes with high computational cost, which is a burden for model applications. There are few possible ways to optimize and speed-up it without investing into expensive hardware:

  - **Model pruning** - reduce the number of layers, units of hidden layers, or the dimension of the embeddings.
  - **Quantization** - sacrife the precision by model weights, use lower 16/8-bit precision instead of 32-bit.
  - **Exporting** - *PyTorch* model can be transfered to a more appropriate format or inference engine, for instance *Torchscript*, *ONNX*
  - **Batching** - predict bigger batches of samples instead of individual samples.

First two require fine-tuning and pre-training from scratch respectively, the last one was applied in our model, hence we will optimize the inference time by exporting *alBERT* to *ONNX* or *Torchscript*. Let's investigate the most suitable technique because inference time is extremely important.

### Results
`
q_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

prideti pareto-frontier of accuracy loss/speed-up ratio:
towardsdtasciencespeedup-bert-inference-different approach
`

### Expermenting

`vanilla PyTorch* will be consider as a baseline `


  
  


ideja is https://towardsdatascience.com/an-empirical-approach-to-speedup-your-bert-inference-with-onnx-torchscript-91da336b3a41



## Further improvements
  - It is possible to create a tokenizer that learns to distinguish the most important parts of sequence, especially if some nucleotide base combinations are of interest (for example, hairpins). The fit tokenizer might improve an efficiency of the transformer. [Quicktour for tokenizer creation](https://huggingface.co/quicktour/transformers).
  - In case you want to push the model even further and employ *large BERT* modifications/alternatives, you should expand a dataset to help model train that massive number of parameters.
   
## Appendix
### A1.Saving & Loading Fine-tuned Model
### A2.Optimizer & Learning Rate Scheduler
### A3.Weight Decay
1. Kaip pasirinkti optimezeri ir learning rate
2. batch size su learning rate santykis
3. kas yra svorio kritimas
4.  Hyperparameters
Authors suggest to use *learning rate = x* (saltinis)
Also, to optimize training time, we suggest to test *batch_size = y*, however it is related to learning rate linearly. (saltinis)


* If have more than 1 GPU:
  ```sh
  if torch.cuda.device_count() > 1:
        print("We can use multiple GPUs, ", torch.cuda.device_count())
        model = nn.DataParallel(model)
  ```
## Acknowledgements
----
1. [HuggingFace](https://huggingface.co) :hugs:
2. i paperius apie svorius learning rate ir etc 
3. [PyTorch tutorials](https://pytorch.org/tutorials/) save&load methds, dataloaders, checkpoints and etc
   



