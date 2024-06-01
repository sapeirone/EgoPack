# A Backpack Full of Skills: Egocentric Video Understanding with Diverse Task Perspectives (CVPR 2024)

Simone Alberto Peirone, Francesca Pistilli, Antonio Alliegro, Giuseppe Averta

**Politecnico di Torino**

<center>
<image src="assets/teaser.jpg" width=400>
</center>

This is the official PyTorch implementation of our work "A Backpack Full of Skills: Egocentric Video Understanding with Diverse Task Perspectives", accepted at CVPR 2024.

**Abstract:** Human comprehension of a video stream is naturally broad: in a few instants, we are able to understand what is happening, the relevance and relationship of objects, and forecast what will follow in the near future, everything all at once. We believe that - to effectively transfer such an holistic perception to intelligent machines - an important role is played by learning to correlate concepts and to abstract knowledge coming from different tasks, to synergistically exploit them when learning novel skills. To accomplish this, we seek for a unified approach to video understanding which combines shared temporal modelling of human actions with minimal overhead, to support multiple downstream tasks and enable cooperation when learning novel skills. We then propose EgoPack, a solution that creates a collection of task perspectives that can be carried across downstream tasks and used as a potential source of additional insights, as a backpack of skills that a robot can carry around and use when needed. We demonstrate the effectiveness and efficiency of our approach on four Ego4d benchmarks, outperforming current state-of-the-art methods.


## Overview
This architecture of EgoPack is structured around a two phase training. First, a multi-task model is trained on a subset of the tasks, e.g., AR, LTA and PNR. Then, the multi-task model is used to bootstrap the EgoPack architecture to learn a novel task, e.g., OSCC.


### Getting started
1. Download all submodules of this repository
```
git submodule update --init --recursive
```

2. Create a conda environment for the project with all the dependendencies listed in the `environment.yaml` file.
```
conda env create -f  environment.yaml
conda activate egopack-env
```

3. Download the Ego4D annotations and Omnivore features from [https://ego4d-data.org/](https://ego4d-data.org/) under the `data/` directory:
   - `data/ego4d/raw/annotations/v1`: `*.json` and `*.csv` annotations from Ego4D.
   - `data/ego4d/raw/features/omnivore_video_swinl`: Omnivore features as `*.pt` files.

4. Some example scripts are provided in the `experiments/` directory and can be executed using the `wandb sweep path/to/config.yaml` command.

#### WandB Integration
The code is organized to heavily rely on [WandB](https://wandb.ai) to run experiments and save checkpoints. In particular, experiments are defined as WandB sweeps with the random seed as one of the experiment's parameter. By doing so, you can easily group experiment by name on the WandB dashboard and evaluate the metrics on the average of three different runs, to obtain more consistent results.

**NOTE:** *A refactored version of this code, with more experiment configs and no dependency on WandB will be released soon (no timeline yet).*

## Cite Us

```
@inproceedings{peirone2024backpack,
    title={A Backpack Full of Skills: Egocentric Video Understanding with Diverse Task Perspectives}, 
    author={Simone Alberto Peirone and Francesca Pistilli and Antonio Alliegro and Giuseppe Averta},
    year={2024},
    booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition}
}
```