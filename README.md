# GWG_release - SYDE 675 Analysis
Official release of code for 
["Oops I Took A Gradient: Scalable Sampling for Discrete Distributions"](https://arxiv.org/abs/2102.04509) 
which has been accepted for a long presentation to ICML 2021. 

The paper is by [Will Grathwohl](http://www.cs.toronto.edu/~wgrathwohl/), Kevin Swersky, Milad Hashemi, 
[David Duvenaud](http://www.cs.toronto.edu/~duvenaud/), and [Chris Maddison](https://www.cs.toronto.edu/~cmaddis/)

I have adapted this code for the purposes of my SYDE 675 - Pattern Recogntion Final Project.
My contributions include protein_analysis_pipeline.py, which imports and/or repurposes many functions written by the 
original authors.
I also make significant use of the evcouplings python package (which is based on 
[this](https://doi.org/10.1371/journal.pone.0028766) paper by Marks et al.) in my train_logreg file, which generates 
features for my own choice of original model/feature combination.

# Requirements
This work will require several python modules to work properly, which the original authors did not include. Hence, I 
generated a requirements.txt. After cloning the repository and creating a virtualenv if desired:
```
cd GWG_release
pip install -r requirements.txt
```
If you want to test the pipeline on an arbitrary protein, you will have to download hmmer as well as a sequence database 
in order to generate MSAs. It is expected by default in a databases folder in this repo and is >32GB compressed:
```
sudo apt install hmmer
mkdir databases
cd databases
wget https://ftp.uniprot.org/pub/databases/uniprot/current_release/uniref/uniref90/uniref90.fasta.gz
gzip -d uniref90.fasta.gz
```
# Training Data
It is not necessary to have the raw (multiple sequence alignment) training data for the Logistic Regression 
as the features have been precomputed and stored. However, it can be downloaded like this:
```
wget http://bioinfadmin.cs.ucl.ac.uk/downloads/contact_pred_datasets/dc_train
tar -xzf dc_train
```
It is sourced from the following paper: 
[Jones DT and Kandathil SM (2018). High precision in protein contact prediction using fully convolutional neural networks and minimal sequence features. Bioinformatics 34(19): 3308-3315.](https://github.com/psipred/DeepCov)
Contact maps are generated independently of this, and so only the aln folder is needed.
Generating features from the training data is an enormous task, and so pickles with pre-extracted features have been provided in the data folder.

# Experimenting with Code
protein_analysis_pipeline.py trains a Dense Potts model for contact prediction, using the Gibbs-with-Gradients algorithm
to sample the model, accelerating training. It should be possible to train a model for an arbitrary experimentally
solved structure from the pdb, provided you have the four-character code and chain corresponding to an entry in the
Protein DataBank: https://www.rcsb.org/
```
python protein_analysis_pipeline.py --pdb_code 1F88 --chain A  --save_dir ./1F88_OPSD_BOVIN_gwg
```
The train_logreg.py file will use a pickle to recieve the input training data by default. You can just call:
```
python train_logreg.py --save_dir ./logreg_test
```
To test this model.
Any files in this repo not mentioned above were generated by the original authors.
