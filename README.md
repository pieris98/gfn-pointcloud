# Diffusion models for amortized inference

Official repository for the paper:

[On diffusion models for amortized inference: Benchmarking and improving stochastic control and sampling](https://arxiv.org/abs/2402.05098)


This repository is divided into two parts, regarding sampling from energies experiments (unconditional GFNs) and the VAE experiment (conditional GFNs).

## Sampling from unknown energies

Firstly, please go to **energy_sampling** directory:

```
cd energy_sampling
```

In order to run the experiment, you should choose one of the following, implemented energy functions:

- **25gmm**
- **hard_funnel**
- **many_well**
- additionally, you can also select **9gmm** or **easy_funnel**

### Exemplary commands to run experiments
Below are commands to reproduce some of the results on **Manywell** with PIS and GFlowNet models as an
example, showing the hyperparameters:

#### GFlowNet TB:
```
python train.py
--t_scale 1. --energy many_well --pis_architectures --zero_init --clipping
--mode_fwd tb --lr_policy 1e-3 --lr_flow 1e-1
```

#### GFlowNet TB + Expl.:
```
python train.py
--t_scale 1. --energy many_well --pis_architectures --zero_init --clipping
--mode_fwd tb --lr_policy 1e-3 --lr_flow 1e-1
--exploratory --exploration_wd --exploration_factor 0.2
```

#### GFlowNet VarGrad + Expl.:
```
python train.py
--t_scale 1. --energy many_well --pis_architectures --zero_init --clipping
--mode_fwd tb-avg --lr_policy 1e-3 --lr_flow 1e-1
--exploratory --exploration_wd --exploration_factor 0.2
```

#### GFlowNet FL-SubTB:
```
python train.py
--t_scale 1. --energy many_well --pis_architectures --zero_init --clipping
--mode_fwd subtb --lr_policy 1e-3 --lr_flow 1e-2
--partial_energy --conditional_flow_model
```

#### GFlowNet TB + Expl. + LS:
```
python train.py
--t_scale 1. --energy many_well --pis_architectures --zero_init --clipping
--mode_fwd tb --lr_policy 1e-3 --lr_back 1e-3 --lr_flow 1e-1
--exploratory --exploration_wd --exploration_factor 0.1
--both_ways --local_search
--buffer_size 600000 --prioritized rank --rank_weight 0.01
--ld_step 0.1 --ld_schedule --target_acceptance_rate 0.574
```

#### GFlowNet TB + Expl. + LP:
```
python train.py
--t_scale 1. --energy many_well --pis_architectures --zero_init --clipping
--mode_fwd tb --lr_policy 1e-3 --lr_flow 1e-1
--exploratory --exploration_wd --exploration_factor 0.2
--langevin --epochs 10000
```
#### PIS:
For PIS, please use the following arguments:
```
--mode_fwd pis --lr_policy 1e-3
```

#### PIS + Langevin:
For PIS + Langevin, please use the following arguments:
```
--mode_fwd pis --lr_policy 1e-3  --langevin
```


## VAE experiment

Firstly, please go to **vae** directory:

```
cd vae
```

For pretraining the VAE model, please run the following command:

```
python energies/vae.py
```

Then, in **energies/vae_energy.py**, please set the path to your newly pretrained VAE model.

```
_VAE_MODEL_PATH = '<path to a pretrained VAE model>'
```

### Exemplary commands to run experiments

Below are commands to reproduce some of the results on **VAE** with GFlowNet models as an
example, showing the hyperparameters:

#### GFlowNet TB + Expl. + LS:
```
python train.py
--energy vae --pis_architectures --zero_init --clipping
--mode_fwd cond-tb-avg --mode_bwd cond-tb-avg --repeats 5
--lr_policy 1e-3 --lr_flow 1e-1 --lr_back 1e-3
--exploratory --exploration_wd --exploration_factor 0.1
--both_ways --local_search
--max_iter_ls 500 --burn_in 200
--buffer_size 90000 --prioritized rank --rank_weight 0.01
--ld_step 0.001 --ld_schedule --target_acceptance_rate 0.574
```

#### GFlowNet TB + Expl. + LP + LS:
```
python train.py
--energy vae --pis_architectures --zero_init --clipping
--mode_fwd cond-tb-avg --mode_bwd cond-tb-avg --repeats 5
--lr_policy 1e-3 --lr_flow 1e-1
--lgv_clip 1e2 --gfn_clip 1e4 --epochs 10000
--exploratory --exploration_wd --exploration_factor 0.1
--both_ways --local_search
--lr_back 1e-3 --max_iter_ls 500 --burn_in 200
--buffer_size 90000 --prioritized rank --rank_weight 0.01
--langevin
--ld_step 0.001 --ld_schedule --target_acceptance_rate 0.574
```


## References
<details>
<summary>
If you find this code useful in your work, please consider citing our paper (expand for BibTeX).
</summary>

```bibtex
@article{sendera2024diffusion,
    title={On diffusion models for amortized inference: Benchmarking and improving stochastic control and sampling},
    author={Sendera, Marcin and Kim, Minsu and Mittal, Sarthak and Lemos, Pablo and Scimeca, Luca and {Rector-Brooks}, Jarrid and Adam, Alexandre and Bengio, Yoshua and Malkin, Nikolay},
    year={2024},
    journal={arXiv preprint arXiv:2402.05098}
}
```
</details>
