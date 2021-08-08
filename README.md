
### To-do
- [ ] Pritaikyti NN modelio outputa (seq1,seq2,label)
- [ ] Pradinis pairing turi issisakoti i dvi puses: pirmas sukurti dataset modelio traininimui, antra atgrupuoti N% geriausiu aptameru GA
- [ ] Ka daryti su abejotinomis sekomis is modelio outputo? t.y. 0.4-0.6 intervalo
- [ ] Prideti iteraciju sekima GA procesui 
- [ ] Prideti default vertes funkcijose
- [ ] Perrasyti preprocesinga GA foldery
- [ ] Paieskoti ar crossoveriui skilimas buna visiskai random, ar tarkim dazniau vidury?
- [ ] Pasalinti harcodintus parametrus
- [ ] Kaip reiktu fine-tuninti modeli
- [ ] Leisti pairing python scriptui priimti dataset kaip inputa (-d)


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
    <a href="https://github.com/"><strong>Explore Wiki »</strong></a>
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
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
----

Prideti keleta sakiniu apie pacius aptamerus.

Sakinys apie SELEX (requires the establishment of suitable protocols and can be labor and cost intensive.)

Therefore the 2015 iGEM Heidelberg team developed a aptamer design tool called MAWS (Making Aptamers Without SELEX) which tries to predict aptamers by progressive selection of appropriate bases using an entropic criterion. 


Here's what have been implemented:
* Iverciu skaiciavimas
* Transformerio pasitelkimas paspartinti procesa
* gal dar kazkas

Of course, no one model will serve all projects since your needs may be different. So transformed model should be pre-trained on your aptamer dataset.



<!-- GETTING STARTED -->
## Getting Started
----
This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites
----


### Installation
----
1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```JS
   const API_KEY = 'ENTER YOUR API';
   ```



<!-- USAGE EXAMPLES -->
## Usage
----
Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_



<!-- CONTRIBUTING -->
## Contributing
----
Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/CuteAptamer`)
3. Commit your Changes (`git commit -m 'Add my feature'`)
4. Push to the Branch (`git push origin feature/CuteAptamer`)
5. Open a Pull Request



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
* [Img Shields](https://shields.io)





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
