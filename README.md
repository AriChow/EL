# EL
Code for emergent symbolic classification

## Installation and setup

```
git clone https://github.com/AriChow/EL.git
pip install -e EL/
```


## Download data
Download CheXpert data from here
```
https://stanfordmlgroup.github.io/competitions/chexpert/
``` 

## Setup
* Update `EL/EL/CONSTS.py` with the locations on your machine.
* Run `EL/EL/annotations.py` to setup training and validation data.
* Update paths in `EL/EL/annotations.py` to obtain test annotations.

## Training
* Train baseline model using `EL/EL/runs/train_chexpert.py`
* Train EL model using `EL/EL/runs/train_chexpert_EL.py`

## Testing
* Test baseline model using `EL/EL/runs/train_chexpert.py` by modifying   
data loader and commenting out training loop.
* Test EL model using `EL/EL/runs/test_chexpert.py`
