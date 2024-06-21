# Candidate KKLT de Sitter vacua

This repository stores the data for the candidate de Sitter vacua obtained in [ArXiv:2406.13751](https://arxiv.org/abs/2406.13751). It also contains Python scripts and notebooks to validate these solutions and reproduce figures from the paper.


## Summary of results

In [ArXiv:2406.13751](https://arxiv.org/abs/2406.13751), we find compactifications of type IIB string theory that are candidate de Sitter vacua at leading order in the $\alpha^\prime$ and $g_s$ expansions of the form envisioned by Kachru, Kallosh, Linde, and Trivedi in [ArXiv:hep-th/0301240](https://arxiv.org/abs/hep-th/0301240). This repository stores the relevant information to compute both the Kähler potential and superpotential for the leading order EFT as defined in section 2 of [ArXiv:2406.13751](https://arxiv.org/abs/2406.13751). Additionally, this repository contains data for a landscape of supersymmetric and non-supersymmetric anti-de Sitter vacua with Klebanov-Strassler throats.

<br>

<p align="center">
  <img src="/images/dS_examples/Example_4/aule_uplift.png" width="450">
</p>


<br>

<br>


## Working with this repository

### Reading the data

To get started, we recommend using the notebook [`reading_data.ipynb`](/notebooks/reading_data.ipynb). We summarize the data structure for the files in [`data`](./data/) and the python scripts that can be found in [`code`](./code/). We also provide a small demo on how to use our data to interface with [CYTools](https://cy.tools) and to compute quantities in the leading order EFT in the notebook [`working_with_examples.ipynb`](/notebooks/working_with_examples.ipynb).

> [!IMPORTANT]
> The code makes use of basic functions from [CYTools](https://cy.tools). For help with the installation, please check out the [documentation](https://cy.tools/docs/getting-started/) or reach out to us.


> [!TIP]
> Note that our code includes the Gopakumar-Vafa invariants for curves used in our computations. To compute additional invariants, we recommend using the publically available package [cygv](https://github.com/ariostas/cygv).





### Code base and validation

We provide Python code in the folder [`code`](./code/) which interfaces to some extent with [CYTools](https://cy.tools). Further, we include validation notebooks in the folder [`notebooks`](/notebooks/) to reproduce the examples presented in the paper. To this end, we implemented a largely independent collection of functions that has **not** been used to produce these solutions. This allows users **to verify these solutions independently**.



### Validation of de Sitter solutions

To validate the five candidate de Sitter solutions from section 5 in the paper, we provide the notebook [`validation_dS.ipynb`](/notebooks/validation_dS.ipynb). The additional 25 de Sitter solutions mentioned in sections 5.4 and 5.5 can also be validated in [validation_extra_dS.ipynb](/notebooks/validation_extra_dS.ipynb).

> [!WARNING]  
> It can be difficult to reproduce the mass spectrum stated in the paper using the finite differences method. We recommend implementing the formulas in appendix D of our paper [ArXiv:2406.13751](https://arxiv.org/abs/2406.13751).


> [!NOTE]
> We intend to upload further candidate de Sitter solutions etc. to this repository in future data releases.


### Validation of further examples

Further, the non-SUSY AdS solutions presented in appendix C of [ArXiv:2406.13751](https://arxiv.org/abs/2406.13751) can be verified by using the notebooks [`validation_SUSY_AdS.ipynb`](/notebooks/validation_SUSY_AdS.ipynb).


## Contact 

For questions or feedback, please get in touch: <as3475@cornell.edu> or <a.schachner@lmu.de>.


## Reference

If you use this database for future publications, please cite

```
@article{McAllister:2024lnt,
    author = "McAllister, Liam and Moritz, Jakob and Nally, Richard and Schachner, Andreas",
    title = "{Candidate de Sitter Vacua}",
    eprint = "2406.13751",
    archivePrefix = "arXiv",
    primaryClass = "hep-th",
    reportNumber = "CERN-TH-2024-090",
    month = "6",
    year = "2024"
}
```
