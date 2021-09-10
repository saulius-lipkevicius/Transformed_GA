<!-- PROJECT SHIELDS -->
<!--
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]



<!-- PROJECT LOGO -->
<br />
<p align="center">
    <img src="images/logo.png" alt="Logo" width="160" height="160">
  </a>

  <h3 align="center">Transformers Enhanced Aptamer Design Software </h3>

  <p align="center">
    An jumpstart to fit aptamers!
    <br />
    <a href="https://igem2021.vilnius.com/"><strong>Explore Wiki »</strong></a>
    <br />
    <br />
    <a href="https://github.com/">Create Issue</a>

  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#motivation">Motivation</a>
    </li>
    <li>
      <a href="#model-dataflow">Model Dataflow</a>
    </li>
    <li><a href="#results">Results</a></li>
    <li><a href="#getting-starter">Getting Started</a></li>
        <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## Motivation
----
Our team this year decided to create an aptamer-based detection method to diagnose amebiasis disease caused by <em>Entamoeba histolytica</em>. Nevertheless, SELEX (Systematic evolution of ligands by exponential enrichment) was chosen as the main approach used to find aptamers for the protein target indicating the presence of <em>E. histolytica.</em>  

Finding a suitable aptamer by the well-established SELEX method requires to set up the appropriate protocols, and might be a laborious and costly procedure. Keeping these reasons in mind, we started to look for <em>in silico</em> approaches for aptamer generation. After studying existing literature resources, we found methods like M.A.W.S. (Making aptamers without SELEX), which was implemented by Heidelberg iGEM 2015 team. Based on this approach, we released an updated version that is described in the [Software page](https://2021.igem.org/Team:Vilnius-Lithuania/Software).

We decided to take a step further and apply a novel transformer-based neural network model  combined with a genetic algorithm to make aptamer generation <em>in silico</em> a more resource-efficient process that has the higher potential to output an affine aptamer sequence. The key part of the model is that it has a property of transfer learning that lets anyone fine-tune the model almost instantly for modified tasks.



### Model Dataflow
-----
Initially, N random aptamer sequences are generated employing ELBAScore software. Following it up, data must be specifically preprocessed to contain a pair of aptamers with a binary label that determines if the second sequence is more fit (1) or not (0).

The dataset of paired sequences is obtained by comparing every aptamer in-between by fitness score, which is computed with the former software.
Later on, the number of classification classes labels are balanced by flipping aptamers places for BERT model to learn both classes equally.

Many transformer-based models could fit this task, however Albert model was chosen because of its state of the Art performance with fewer parameters than the threshold BERT model, which takes 4-5 times less time to train, saving days of expensive GPU runtime. Working with huge datasets like ours time is the main reasoning for choosing Albert. 

Another significant part of the model is the genetic algorithm (GA) that produces new sequences at every iteration by well-known breeding and mutation steps. Additionally, GAs probabilistic model helped to determine the convergence and how many iterations the process requires to produce the final aptamer list which consists of N aptamers to be investigated further. 

Lastly, the sequences of the final iteration are analyzed and compared by ELBAScore. Furthermore the top 10 % of total will be reevaluated in the lab.


## Results
-----
Two separate models were created for protein targets albumin and EhPPDK. Here the transfer learning helped out - we had to train a model only on an albumin dataset for 2.5 days on 1 GPU and later on only to fine-tune the same Albert model with EhPPDK protein target dataset. This approach saved us some time, since it took ~3 hours for the model to relearn positional embedding to inference partially different data. 

The initial model itself was trained on 1500 different aptamer sequences data from ELBAScore, which formed 1,124,250 pairs with binary labels, 60% of it was used for training matter, 20 % for validation, and the remaining 20% for testing. To inference a new population of aptamers Albert takes approximately 5 minutes. [metrikos] + [top aptameru iverciai su ELBALite, kokia dalis nukeliavo i labe] + [gal dar kazkokius iteracinius/tarpinius duomenis] + [pabrezti kaip efektino] + [distribucijos issemimas]

<figure><img src="images/Albumin Model Train.png" alt="" width="40%" height="300" title="Albumin Model Training and Validation Losses"><img src="images/Mixed Model Train.png" alt="" width="40%" height="300">
<figcaption align = "center"><b>Target protein Albumin and EhPPDK-Albumin model training and validation losses</b></figcaption></figure>

<figure><img src="images/ROC Albumin Model.png" alt="" width="45%" height="300" title="Albumin Model Training and Validation Losses" align="center"><img src="images/ROC Mixed Model.png" alt="" width="45%" height="300" align="center>
<figcaption align = "center"><b>Target protein Albumin and EhPPDK-Albumin model ROC curves</b></figcaption></figure>

<figure><img src="images/Albumin Model Confusion Matrix.png" alt="" width="45%" height="350" title="Albumin Model Training and Validation Losses"><img src="images/Mixed Model Confusion Matrix.png" alt="" width="45%" height="350">
<figcaption align = "center"><b>Target protein Albumin and EhPPDK-Albumin model confusion matrices</b></figcaption></figure>


<!-- GETTING STARTED -->
## Getting Started
----
This is an example of how you may give instructions on setting up your project locally. In order to run the model locally, follow these simple example steps.

### Prerequisites & Installation
----
To quickly install all packages required for algorithm run:
```
pip install requirements.txt
```

In case you are running on cloud there is a perfect [tutorial](https://medium.com/analytics-vidhya/install-cuda-11-2-cudnn-8-1-0-and-python-3-9-on-rtx3090-for-deep-learning-fcf96c95f7a1) on how to install every dependency you might need to train a deep learning model. These dependencies include Cuda, CudaNN, and PyTorch. However, if you have no access to cloud GPU instances, we strongly suggest to utilize [Google Colab](https://link-url-here.org).


<!-- USAGE EXAMPLES -->
## Usage
----
Project can be used in two ways. In case you have the same type of dataset and the task to work on, the model is shared in the AI community [HuggingFace](https://huggingface.co/models) by name "VilniusIGEM2021/albert-base-aptamers". One command to rule them all and inference as with usual transformer-based model:

```
model = AutoModel.from_pretrained('VilniusIGEM2021/albert-base-aptamers')
```

More information related to this flow can be found in [HuggingFace/Transformers](https://huggingface.co/transformers/). 

Otherwise, if task differs, for example in case of the longer aptamer sequences or it is required to change the task from classification to sequence generation, then you have to run the process described in `model` folder with changed initial `albert-case-v2` model to `VilniusIGEM2021/albert-base-aptamers`.

_For more in-depth ALBERT model description and explanation, please refer to the [ALBERT Documentation.](https://github.com/saulius-lipkevicius/GA_Transformer/tree/main/model)_


## V2.0 Optimization

  - Optimizing number of aptamers taken in every sequence by common derivate calculation:
  - Optimizing with exporting to ONNX
  - Otimizing by diminishing accuracy to INT8
  - Change structure of comparing
  - Change algorithm flow


<!-- CONTRIBUTING -->
## Contributing
----
Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**. 

In order to contribute, please follow the steps below:

1. Fork the project
2. Create your feature branch (`git checkout -b feature/CuteAptamer`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push to the branch (`git push origin feature/CuteAptamer`)
5. Open a pull request

###  Contributing to HuggingFace
Any contribution to the AI community HuggingFacce community is super valuable, find more information in [HuggingFace/Contributing.](https://huggingface.co/transformers/contributing.html)



###  Suggestion for future improvements

  1. Model code can be rewriten to TensorFlow.
  2. Different transformer-based models can be tried out, for instance, RoBerta, GPT-2 and so on.
  3. To make model more precise, a model embedding 3 classes instead of 2 could be considered. The third class could stand for the unknown relationship between a pair of aptamers.
   


<!-- LICENSE -->
## License
----
Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact
----
Saulius Lipkevičius - sauliuslipkevicius@gmail.com


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
----

* [IGEM Heidelberg 2015 team](http://2015.igem.org/Team:Heidelberg)




<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[product-screenshot]: images/screenshot.png
