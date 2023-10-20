# BoB-OOD-Classification

This repository is the official implementation of <strong>Out-of-Distribution Image Classification</strong> task in the [*Battle of the Backbones: A Large-Scale Comparison of Pretrained Models across Computer Vision Tasks*](https://github.com/hsouri/Battle-of-the-Backbones).

## Dependencies

Version Control of Python libraries in environment.yml file. To create a virtual environment:
```bash
conda env create -f environment.yml
```
### Classification OOD experiments

#### Dependencies:

Install tlllib
```
python3 -m pip install -i https://test.pypi.org/simple/ tllib==0.4
```

#### Instructions
We include shell scripts to benchmark OOD generalization performance for image classification: for robustness to style and structure variations on ImageNet variants (ImageNet Sketch, Renditions, Adversarial, and V2), and synthetic-to-real generalization on VisDA2017. 

To evaluate on ImageNet variants, run:

```
bash eval_classifier_ood.sh
```

To train on VisDA (syn), and evaluate on VisDA (real), run:
```
bash eval_classifier_ood.sh
```

Update the following variables as appropriate:
```
BACKBONE, VALID_LABELS_INA, VALID_LABELS_INR, CHECKPOINT, DATA_DIR
```
