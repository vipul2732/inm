# Integartive Network Modeling

Integrative network modeling (INM) is a method and framework (library) for modeling biological interaction networks. The INM library supports the modeling of protein interactions networks based on AP-MS data.
INM can:
- Predict direct protein-protein interaction (PPI) networks for all pairs of protein types (matrix model) on the order of hundreds or perhaps thousands of nodes
- Predict condition specific PPI networks (e.g., gene deletion, chemical perturbation, cell-type)
- Predict a distribution over networks to estimate both the average network and ascociated uncertainty
- Has a scoring funciton that may be extended to other types of information that may inform PPI networks (e.g, deep learning based PPI prediction, proximity labeling) 


Modeling proceeds in 5 stages: (i) gathering AP-MS protein spectral counts under multiple conditions, (ii) representing a model where nodes represent protein types
and edges represent a protein interaction between at least one pair of molecules of the corresponding types, (iii) a scoring function
that scores the agreement between AP-MS spectral counts and a network model, (iv) sampling alternative configurations of edges using an MCMC scheme,
(v) analysis of the output sample of network models.

inm includes an example system and benchmarks against Protein Data Bank and Humap 2.0 derived interactions.  

## Installation

To clone the repository use the following command
```bash
git clone https://github.com/ajipalar/inm.git
```

Create and activate the conda environment
```bash
cd inm
conda env create -f environment.yml
conda activate py39
```

## Run tests
```bash
python3 -m unittest discover -s tests
```

 INM is a part of the Integrative Modeling Platform ([IMP](https://github.com/salilab/imp/)) module.
