

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
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The BERT
1. Apie BERT:



2. Kodel pasirinkta alBERT variantas
3. Advantages
4. kaip NLP siejasi su sekomis
## Getting Started
----
### Setup
1. Galimi variantai
2. kaip susirinkti dependancies ir pan
3. akip pateikti duomenis

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Tokenization & input Formatting
1. Prideti nuotrauka proceso
2. imesti pavyzdi tokenizavimo
3. kaip tai gali itakoti rezultatus
4. kaip tai skiriasi nuo standartinio modelio pritaikymo ir kokie papildomi tokenai naudojami (gali buti)
5. (gal prie training) kokio ilgio sekas dedame, kas yra attention mask
6. kas ivyksta su ilgesnem sekom, apie batchinima
### Train
1. train,test,val splitas
2. apie in-built torch iteratoriu duomenims
3. bertForSequenceClassification atvejis
4. layerius ir kitus galimus modelius
5. head'us
6. optimizer ir learning rate (gal dar papildomas saltinis apie linear growth learning rate)
7. in general training loopas + keletas image is proceso o gal ir
<!-- USAGE EXAMPLES -->
## Usage
----
Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_
## Model performance
1. iverciai ir grafikeliai training/val loss
2. test data ivertinimas (gal paieskoti kitu iverciu ) + confuson matrica
## Conclusion
----

## Further improvements
1. Traininti tokenizeri paciam, ypac jeigu sekoje yra dominanciu strukturu, kaip hairpinai.
2. isplesti dataseta, nes jis auga eksponentiskai, gal net praretinti ypac panasaus ivercio aptamerus
3. Galima prasitestuoti beveik visas BERT atmainas, ypac verta atkreipti demesi i large modelius + resursu santykis su accuracy
4. padaryti modeli prieinama huggingface
### Fine-tuning
## Appendix
### A1.kazkas
### A2 kitas
### A3 DAR KITS
1. Kaip pasirinkti optimezeri ir learning rate
2. batch size su learning rate santykis
3. kas yra svorio kritimas
4. ar laudinti modeli, t.y. sioje skiltyje???

## Acknowledgements
----
 
1. huggingface quicktour tokenizeriui + modeliui [Img Shields](https://huggingface.com) :smile:
2. i paperius apie svorius learning rate ir etc 

