# Apollo: <u>A</u> <u>Po</u>steriori <u>L</u>abe<u>l</u>-<u>O</u>nly Membership Inference Attack Towards Machine Unlearning


## 1. Requirements

```
PyTorch >= 2.5.0
cuda    >= 11.8.0
```

For a full list of the environments used in our execution, see `requirements.txt`.

## 2. Usage

### 2.1 Training

To train the **target model**, run:

```
python main_pretrain.py --model {architecture} --dataset {dataset} --num_classes {n} --input_size {image size, e.g., 3 32 32 for CIFAR-10 and CIFAR-100} --batch_size {batch size} --lr {learning rate} --opt {optimizer, supporting SGD and adamw} --epochs {training epochs} --size_train {size of the training set}
```

### 2.2 Unlearning

Subsequently, to **unlearn** samples in the training set, run:

```
python main_unlearn.py --model {architecture} --dataset {dataset} --num_classes {n} --input_size {image size, e.g., 3 32 32 for CIFAR-10 and CIFAR-100} --batch_size {batch size} --size_train {size of the training set} --unlearn {unlearning algorithm} --forget_perc {percentage of unlearned set}
```

Ensure that the information provided is identical to the pre-trained model. The unlearning algorithm will save:

* the unlearned model $\theta_u$ as `unlearn.pth.tar`
* a dictionary of the indexes of the training set $D$, unlearned set $D_u$, retain set $D_r$ as `data_split.pkl`
* evaluation results of the unlearned model as `eval_results.pkl`
* arguments for the unlearning process as `unlearn_args.pkl`

For *class-wise unlearning*, substitute `--forget_perc {percentage of unlearned set}` with `--forget_class {class to be unlearned}`.

To train shadow models for `Apollo` and `U-LiRA`, run:

```
python shadow_train.py --model {architecture} --dataset {dataset} --num_classes {n} --input_size {image size, e.g., 3 32 32 for CIFAR-10 and CIFAR-100} --batch_size {batch size} --lr {learning rate} --opt {optimizer, supporting SGD and adamw} --epochs {training epochs} --size_shadow {size of the surrogate set for each shadow model} --num_shadow {numbers of shadow model to train} --split {how the shadow model samples its training data, "full" or "limited"}
```

Which will save each shadow model as `i.pth.tar` under the same folder, along with a `data_split.pkl` containing a dictionary of the indexes of each shadow set. For additional information on the sampling of shadow sets, we refer to Appendix A of our paper.

### 2.3 Attack

We implement our proposed `Apollo` attack algorithm as well as two existing algorithms, `U-MIA` by [Kurmanji et al.](https://openreview.net/forum?id=OveBaTtUAT) and `U-LiRA` by [Hayes et al.](https://arxiv.org/abs/2403.01218), to perform an attack on the targeted unlearned model, run:

```
python attack.py --model {target model architecture} --dataset {dataset} --num_classes {n} --input_size {image size, e.g., 3 32 32 for CIFAR-10 and CIFAR-100} --target_path {path to the target model folder} --num_shadow {number of shadow models to be utilized} --shadow_model {shadow model architecture} --shadow_path {path to the shadow models folder} --atk {attack type, use "Apollo", "Apollo_Offline", "ULiRA" or "UMIA"} --N {number of targets to be attacked in the unlearned and test set}
```

For the online version of `Apollo` and `U-LiRA`, the attack will try to unlearn partials of the shadow models first. This requires substantial time on the first run, however, the resulting unlearned shadow models will be stored and can be reused.

## 3. Acknowledgements

Many of the code used in this repository are forked from the [code implementations](https://github.com/K1nght/Unified-Unlearning-w-Remain-Geometry) of [Huang et al.](https://arxiv.org/abs/2409.19732), as noted in our paper. We thank the authors for making their code public.