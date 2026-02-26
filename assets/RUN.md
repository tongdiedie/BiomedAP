# Training and Evaluation

We provide bash scripts in [scripts/](../scripts) for each technique including prompt learning and other few-shot adaptation techniques.
Make sure to configure the dataset paths in environment variable `DATA` and run the commands from the main directory `BiomedDPT/`.
Below we provide training and evaluation instructions for Biomed-DPT. The same instructions applies for all other techniques.


### Training time and compute
We train BiomedCoOp on each dataset with a batch size of 4 using a **three** NVIDIA 4090 GPU. Currently. You will have to specify the GPU number that you want to use.

## Biomed-DPT

#### (1) Few-shot evaluation setting

The default training settings are provided in the config files at `configs/trainers/BiomedDPT/few_shot`. All hyper-parameters can be modified using this config file.

Below, we provide instructions to train Biomed-DPT  on any dataset. 

```bash
# All possible dataset values include [btmri, busi, chmnist, covid, ctkidney, dermamnist, kneexray, kvasir, lungcolon, octmnist, retina]

# CLIP Models include [CLIP, PubMedCLIP, PMCCLIP, BiomedCLIP]

# trains and evaluates in a few-shot setting on all 3 seeds
CUDA_VISIBLE_DEVICES=<GPU number> bash scripts/biomedcoop/few_shot.sh <data directory> <dataset> <nb of shots> <clip model to use>
# Example on BTMRI using 16 shots and the BiomedCLIP model on GPU 0
CUDA_VISIBLE_DEVICES=0 bash scripts/biomedcoop/few_shot.sh data btmri 16 BiomedCLIP
```

you can also use few_shot_coop.py to train all coop-based methods on any dataset. 

```
python few_shot_coop.py
```

#### x # Instructions borrowed from https://github.com/KaiyangZhou/Dassl.pytorch#installationcd Dassl.pytorch/​# Install dependenciespip install -r requirements.txt​# Install this librarypython setup.py developcd ..bash

Once the above trainings and evaluations are completed, the `output/` directory should have the following structure:

```
output
|–– btmri/
|   |–– shots_16/
|   |   |–– BiomedDPT_BiomedCLIP/
|   |   |   |–– nctx4_cscFalse_ctpend/
|   |   |   |   |–– seed1/
|   |   |   |   |–– seed2/
|   |   |   |   |–– seed3/
```

Now use the script `show.py` and run the commands below to show results:
```bash
# prints averaged results
python show.py
|––show_base_to_new() : show base2new results
|––**show_few_shot_coop()**: show the results of all few-shot coop-based methods 
|––show_zeroshot() : show the results of all zero-shot VLMs 
|––show_few_shot_adapter() : show the results of all few-shot adapter-based methods 
|––show_base_model() : show the results of few-shot adapter-based methods based all VLMs
```

The above steps can be repeated for other individual datasets.

#### (2) Base-to-Novel class generalization setting

Below, we provide instructions to train Biomed-DPT  on Base-to-Novel class. 

```bash
# All possible dataset values include [btmri, chmnist, covid, ctkidney, dermamnist, kneexray, kvasir, lungcolon, octmnist, retina]

# CLIP Models include [CLIP, PubMedCLIP, PMCCLIP, BiomedCLIP]

# trains and evaluates on base and novel classes
CUDA_VISIBLE_DEVICES=<GPU number> bash scripts/biomedcoop/base2new.sh <data directory> <dataset> <clip model to use>
# Example on BTMRI using the BiomedCLIP model on GPU 0
CUDA_VISIBLE_DEVICES=0 bash scripts/biomedcoop/base2new.sh data btmri BiomedCLIP
```

you can also use base2new.py to train all coop-based methods on Base-to-Novel class. 

```
python base2new.py
```



#### Averaging results over 3 seeds: 

Once the above trainings and evaluations are completed, the `output/` directory should have the following structure:

```
output
|–– base2new/
|   |–– test_new/
|   |   |–– btmri/
|   |   |   |–– shots_16/
|   |   |   |   |–– BiomedDPT_BiomedCLIP/
|   |   |   |   |   |–– nctx4_cscFalse_ctpend/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
|   |–– train_base/
|   |   |–– btmri/
|   |   |   |–– shots_16/
|   |   |   |   |–– BiomedDPT_BiomedCLIP/
|   |   |   |   |   |–– nctx4_cscFalse_ctpend/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
```

Now use the script `show.py` and run the commands below to calculate the averaged results:
```bash
python show.py
|––**show_base_to_new()** : show base2new results
|––show_few_shot_coop(): show the results of all few-shot coop-based methods 
|––show_zeroshot() : show the results of all zero-shot VLMs 
|––show_few_shot_adapter() : show the results of all few-shot adapter-based methods 
|––show_base_model() : show the results of few-shot adapter-based methods based all VLMs
```

The above steps can be repeated for other individual datasets.

#### Training and Evaluating other techniques

For other techniques, we provide their corresponding configs and scripts as follows.

```
configs
|–– datasets/
|–– trainers/
|   |–– BiomedCoOp/
|   |-- BiomedDPT/
|   |–– CLIP_Adapter/
|   |–– CoCoOp/
|   |–– CoOp/
|   |–– KgCoOp/
|   |–– LP/
|   |–– LP2/
|   |–– ProGrad/
|   |–– TiP_Adapter/
|   |–– Zeroshot/
```

```
scripts
|–– biomedcoop/
|–– biomeddpt/
|–– clip_adapter/
|–– cocoop/
|–– coop/
|–– kgcoop/
|–– linear_probe/
|–– linear_probe2/
|–– prograd/
|–– tip_adapter/
|–– zeroshot/
```

Please use the corresponding config and script files and follow the same instructions as provided for BiomedCoOp in order to train and evaluate the other variants. 

#### Acknowledgements
This file for running the methods has been borrowed from [MaPLe's](https://github.com/muzairkhattak/multimodal-prompt-learning/blob/main/docs/RUN.md) official repository.
