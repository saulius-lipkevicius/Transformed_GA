# Content
- [Content](#content)
  - [Introduction](#introduction)
  - [Setup](#setup)
  - [Tokenization & input Formatting](#tokenization--input-formatting)
  - [Train](#train)
  - [Model performance](#model-performance)
  - [## Conclusion](#-conclusion)
  - [Further improvements](#further-improvements)
  - [Appendix](#appendix)
    - [A1.kazkas](#a1kazkas)
    - [A2 kitas](#a2-kitas)
    - [A3 DAR KITS](#a3-dar-kits)
  - [References](#references)
## Introduction
1. Apie BERT
2. Kodel pasirinkta alBERT variantas
3. Advantages
4. kaip NLP siejasi su sekomis
## Setup
1. Galimi variantai
2. kaip susirinkti dependancies ir pan
3. akip pateikti duomenis
## Tokenization & input Formatting
1. Prideti nuotrauka proceso
2. imesti pavyzdi tokenizavimo
3. kaip tai gali itakoti rezultatus
4. kaip tai skiriasi nuo standartinio modelio pritaikymo ir kokie papildomi tokenai naudojami (gali buti)
5. (gal prie training) kokio ilgio sekas dedame, kas yra attention mask
6. kas ivyksta su ilgesnem sekom, apie batchinima
## Train
1. train,test,val splitas
2. apie in-built torch iteratoriu duomenims
3. bertForSequenceClassification atvejis
4. layerius ir kitus galimus modelius
5. head'us
6. optimizer ir learning rate (gal dar papildomas saltinis apie linear growth learning rate)
7. in general training loopas + keletas image is proceso o gal ir
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
## Appendix
### A1.kazkas
### A2 kitas
### A3 DAR KITS
1. Kaip pasirinkti optimezeri ir learning rate
2. batch size su learning rate santykis
3. kas yra svorio kritimas
4. ar laudinti modeli, t.y. sioje skiltyje???

## References
1. huggingface quicktour tokenizeriui + modeliui
2. i paperius apie svorius learning rate ir etc 