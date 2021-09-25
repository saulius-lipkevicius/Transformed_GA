

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
2018 was a breakthrough year for NLP, multiple new models like OpenAI's Open-GTP, Google's BERT allowed researchers to fine-tune existing models to produce state-of-art performance with minimal effort. Almost instatly LSTM was largely replaced by BERT (Bidirectional Encoder Representations from Transformers) was released in late 2018, initialy model was employed to Natural Language Processing (NLP), nowadays it finds new applications in variety of fields: machine translation, document generation, biological sequence analysis, video understanding and etc. 

In this project, we will be using BERT modification a Lite BERT (alBERT) that incorporated parameter-reduction techniques to avoid memory limitations of available software, hence has multiple times less parameters to learn and trains 1.7x faster also archieves slightly worse performance. Multiple IGEM teams have tried to apply well-established deep learning methods like CNN, LSTM to predict some features of biological sequences, however those architectures have gradient flaws that especially reveal itself in long sequences. Lets consider why transformers have an edge over former models.


- [ ] Using a pre-trained network generally makes sense if both tasks or both datasets have something in common.
- [ ]  Instead of training the other neural network from scratch, we “transfer” the learned features.
- [ ] Prideti apie traininimo ilguma ir neefektyvuma kainos atzvilgiu, plius prieinamuma-->the most prominent is the cost of running algorithms
- [ ] Simply put, a pre-trained model is a model created by some one else to solve a similar problem. Instead of building a model from scratch to solve a similar problem, you use the model trained on other problem as a starting point.
- [ ] What is our objective when we train a neural network? We wish to identify the correct weights for the network by multiple forward and backward iterations. By using pre-trained models which have been previously trained on large datasets, we can directly use the weights and architecture obtained and apply the learning on our problem statement. This is known as transfer learning. We “transfer the learning” of the pre-trained model to our specific problem statement.
- [ ] If the problem statement we have at hand is very different from the one on which the pre-trained model was trained – the prediction we would get would be very inaccurate
- [ ] many pre-trained models are available
- [ ] We make modifications in the pre-existing model by fine-tuning the model. Since we assume that the pre-trained network has been trained quite well, we would not want to modify the weights too soon and too much. While modifying we generally use a learning rate smaller than the one used for initially training the model
- [ ] Albert-base model for sequence classifcation mostly has ~82-87% of accuracy which is quite similar for our dataset
- [ ] Ways to Fine tune the model: Feature extraction, Use the Architecture of the pre-trained model, Train some layers while freeze others more info onhttps://www.analyticsvidhya.com/blog/2017/06/transfer-learning-the-art-of-fine-tuning-a-pre-trained-model/
- [ ] Rather than pre-defining or using off-the-shelf hyperparameters, simply tuning the hyperparameters of our model can yield significant improvements over baselines.
- [ ] https://towardsdatascience.com/7-tips-to-choose-the-best-optimizer-47bb9c1219e 
This division is exclusively based on an operational aspect which forces you to manually tune the learning rate in the case of Gradient Descent algorithms while it is automatically adapted in adaptive algorithms
Adam is the best among the adaptive optimizers in most of the cases.
Adam is the best choice in general. Anyway, many recent papers state that SGD can bring to better results if combined with a good learning rate annealing schedule which aims to manage its value during the training.
My suggestion is to first try Adam in any case, because it is more likely to return good results without an advanced fine tuning.
Then, if Adam achieves good results, it could be a good idea to switch on SGD to see what happens.


### Model advantages
---
* **Quick Development**
  - Compared to LSTM, BERT has a property of transfer learning, that means you don't have to train model lower layers from scratch, just to apply *head-layer* that suits your task, to get state-of-art results. 

* **Less Data**
  - Best performing deep learning models have millions of parameters to train, therefore model from scratch requires immense size datasets, a lot of time, and hands to create dataset.
  
* **Better Results**
    - It was shown that simple fine-tuning, by adding one layer on the top of BERT, can archieve state-of-art results with minimal task-specific adjustments and it does not suffer from vanishing/exploding gradient (RNN ilness). As a consequence BERT can handle long sequences.

* **Completely Exhaust GPU resources**
    - RNN, LSTM were hardly parallelizable because of recurrent-like architecture, to avoid issue BERT employed the new [attention link] methodology that lets BERT to fully parallelize computations. *Albert code is written to support multiple GPUs

## Getting Started 
----
### Colab Setup
Google Colaboratory offers free GPUs which is perfect to train large neural networks like alBERTs. To add GPU select on menu:

`Edit -->  Notebook Settings --> Hardware accelerator --> (GPU)`

This Framework has some time and resourses drawbacks if training dataset is huge or *large* alBERT architecture is chosen, hence we used Google Colaboratory Pro to speed up the process.


### Preprocessing and Data Locating
Model input must follow standard norms - **(Sequence1, Sequence2, Label)**. In case you have list of sequences from **M.A.W.S** you have to run it through *Python* script *pairing.py* to generate labeled dataframe for training.

* Pairing aptamers to fit it into *alBERT*
  ```sh
  python pairing.py -d aptamerListCSV
  ```
* Output format
<p align="center">
    <img src="./../images/dataframe.png" alt="dataframe" width="360" height="160">
</a>s


### Dependencies

To use a pre-trained transformer *HuggingFace* :hug: provides API to quickly download and use those on a give dataset. API contains thousands of pretrained models to perform many tasks including all *BERT* modifications, however in our case we employed *alBERT* for classification. More information on *pytorch interface*(https://pypi.org/project/transformers)

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
  - *Special* tokens at the beginning and ending of each sentence.
  - Padding & truncation to a single constant length. (papildyti)
  - Differ *real* tokens from *padding* tokens with attention mask.

### Special Tokens

`[CLS]` - For classification tasks, this token is appended in the beginning of first sentence. The significance of this token appears after all embeddings and produce classifier value - prediction.

`[SEP]` - Appears in the ending of *every* sentence and is given to seperate sentences to help model determine something.

`[PAD]` - Is used to balance every input sequence lenghts.

ITERPTI ARCHITEKTUROS PNG kad butu suprantamiau
<p align="center">
    <img src="./../images/logos.png" alt="Logo" width="160" height="160">
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

In case, dataset consists of varying length aptamers we have to consider two *alBERT* constraints:

* Every aptamer pair must be padded or truncated to a same, fixed length.

* The maximum lump length can't exceed 512 tokens.

 However, keep *max_len* as small as possible since training time approximaly linearly dependent on this parameter. [linkas]


## Training
----
Following good practice, data was divided up in *train*, *test*, *validation* groups with *80%*, *10%*, *10%* percentage of data respectively, refer to *pairing.py*.
Next, an *iterator* for our dataset using *torch DataLoauder* class is created, which helps to save memory compared to simply data looping which stucks whole loaded data to memory.

* train, validation datasets
  ```sh
  train_loader = DataLoader(train_set, batch_size=bs, num_workers=1)
  val_loader = DataLoader(val_set, batch_size=bs, num_workers=1)
  ```
  Machine with GPU has multiple cores, this means that the next batch can already be loaded and ready to go by the time the main process is ready for another batch. This is where the *number_of_workers* comes and speeds up, batches are loaded by workers and queued up in memory.  Optimal number of workers is equal 1.[https://deeplizard.com/learn/video/kWVgvsejXsE]

Check Appendix to check how other hyperparameters can be optimized.
- [ ] Using a pre-trained network generally makes sense if both tasks or both datasets have something in common.


### V2.0 Optimization

  - [ ] Optimizing number of aptamers taken in every sequence by common derivate calculation:
  - [ ] Optimizing with exporting to ONNX
  - [ ] Otimizing by diminishing accuracy to INT8
  - [ ] Change structure of comparing
  - [ ] change algorithm flow
  - [ ] prideti kodel renkames ADAMW optimizer


### Model optimization with ONNX
Transformers and transformers-like achitectures have taken over many sequence related field with de-facto state-of-art performance, however it comes with high computational cost which is a burden for inference, usage of model in applications. There are few possible ways to optimize and speed-up it withoutinvesting into expensive hardware:

  - **Model pruning** - Reduce the number of layers, hidden layers units or the dimension of the embeddings.
  - **Quantization** - Sacrife model weights precision, use lower 16/8-bit precision isntead of 32-bit.
  - **Exporting** - *PyTorch* model can be transfered to more appropiate format or inference engine, for instance *Torchscript*, *ONNX*
  - **Batching** - predict bigger bataches of samples instead of individual samples.

First two requires fine-tuning and pretraining from scratch respectively, the last one was applied in our model, hence we will optimize inference time by exporting *alBERT* to *ONNX* or *Torchscript*. Let's investigate the most suitable technique because inference time is extremely important.

- [ ] BY  https://timdettmers.com/2018/10/17/tpus-vs-gpus-for-transformers-bert/
TPUs are about 32% to 54% faster for training BERT-like models.alternatyva T4 16gb x4 ~ 1k month

### Results
`
q_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

- [ ] prideti pareto-frontier of accuracy loss/speed-up ratio:
towardsdtasciencespeedup-bert-inference-different approach
`

### Expermenting

`vanilla PyTorch* will be consider as a baseline `

- [ ] Suforminti palyginima ONNX ir precision
  ideja is https://towardsdatascience.com/an-empirical-approach-to-speedup-your-bert-inference-with-onnx-torchscript-91da336b3a41



## Further improvements
  - It is possible to create a tokenizer that learns to distinguish the most important parts of sequence, especially if some nucleotide base combinations are of interest, for instance hairpins. Fit tokenizer might improve transformers efficiency. [Quicktour for tokenizer creation](https://huggingface.co/quicktour/transformers).
  - In case you want to push model even further and employ *large BERT* modifications/alternatives, you should expand a dataset to help model train that massive number of parameters.
   
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

- [ ] ADAMW optimizacija:
an optimizer with weight decay fixed that can be used to fine-tuned models
Adam (Kingma & Ba, 2015) [42] is one of the most popular and widely used optimization algorithms and often the go-to optimizer for NLP researchers.
It is often thought that Adam clearly outperforms vanilla stochastic gradient descent (SGD). However, while it converges much faster than SGD, it has been observed that SGD with learning rate annealing slightly outperforms Adam (Wu et al., 2016)
- [ ] How you can train a model on a single or multi GPU server with batches larger than the GPUs memory or when even a single training sample won’t fit (!),


## Acknowledgements
----
1. [HuggingFace](https://huggingface.co) :hugs:
2. i paperius apie svorius learning rate ir etc 
3. [PyTorch tutorials](https://pytorch.org/tutorials/) save&load methds, dataloaders, checkpoints and etc
4.  read more about transfer-learning https://www.analyticsvidhya.com/blog/2017/06/transfer-learning-the-art-of-fine-tuning-a-pre-trained-model/

   



