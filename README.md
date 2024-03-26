# simpleDiffusionPolicy
this project modified from https://github.com/real-stanford/diffusion_policy at Colab (vision).


## Installation
### Simulation
use conda
```bash
conda env create -f conda_env.yaml
```
### train
```bash
conda activate simpledp
python trainDP.py
```
### inference

```bash
## in inference.py file, this parameter need to modify: load_pretrained = True
python inference.py
```