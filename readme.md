## LEMON-LanguagE-ModeL-for-Negative-Sampling-of-Knowledge-Graph-Embeddings

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.6](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)

## Steps for running the repository

Clone the github repository
```
git clone https://github.com/ColdMist/LEMON-LanguagE-ModeL-for-Negative-Sampling-of-Knowledge-Graph-Embeddings.git
```
Obtain the datafiles 
```
coming soon
```
Using appropriate data directory run lm2vec.py 
```
cd utils
python lm2vec.py
```
Example command for running scrips
```
python -u ../codes/run.py --do_train --cuda --do_test --data_path ../data/wn18rr/ --model RotatE --hidden_dim 500 --negative_sample_size 50 --batch_size 512 --gamma 6.0 --adversarial_temperature 0.5 -lr 5e-05 --max_steps 80000 -khop 3 --log_steps 5000 -nclusters 15 --dataset wn18rr -de -save models/RotatE/wn18rr/500/5e-05/512/6.0/50/0.5/80000/15/3/adv -c_low_scores --low_score_collection_threshold 200 -adv
```
