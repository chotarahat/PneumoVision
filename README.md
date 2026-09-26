# PneumoVision

Explainable AI-Based Pneumonia Detection from Chest X-Ray Images.

> Research/education project. This system is **not a medical device** and must not be used for diagnosis or treatment decisions.

## Project goals

1. Train a reproducible pneumonia classifier using PyTorch.
2. Evaluate it with clinically relevant metrics.
3. Add Grad-CAM explanations.
4. Provide a simple Streamlit web interface.
5. Keep the experiment structure suitable for later thesis/publication work.

## Initial model

The first baseline uses transfer learning with `ResNet18` and two classes:

- NORMAL
- PNEUMONIA

PyTorch's official transfer-learning guidance recommends pretrained CNNs as a practical starting point for small/medium image datasets. See the references below.

## Dataset

For the first development baseline, use the public Chest X-Ray Images (Pneumonia) dataset. It contains 5,863 JPEG images in two categories and is derived from pediatric chest radiographs from Guangzhou Women and Children's Medical Center.

Dataset page:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

For the first baseline, use the dataset's provided train and test directories. The validation split created inside the training set is still an image-level split. Do not claim publication-grade generalization from it. The development dataset contains multiple images from patients, so a rigorous study should use patient-level separation when patient identifiers are available and/or a genuinely independent external dataset.

## Repository structure

```text
PneumoVision/
├── app/
│   └── streamlit_app.py
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── models/
├── notebooks/
├── reports/
│   ├── figures/
│   └── metrics/
├── src/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── gradcam.py
├── requirements.txt
├── .gitignore
└── README.md
```

## First milestone

Before adding FastAPI, authentication, databases, or extra diseases:

- [ ] Dataset downloaded
- [ ] Data integrity checked
- [ ] Baseline ResNet18 trained
- [ ] Validation metrics generated
- [ ] Confusion matrix generated
- [ ] ROC-AUC generated
- [ ] Grad-CAM generated
- [ ] Streamlit inference page working
- [ ] Experiment configuration committed
- [ ] Results reproducible from a clean environment

## Later research milestones

- Patient-level splitting
- Class-imbalance analysis
- Calibration
- Sensitivity/specificity and confidence intervals
- Repeated cross-validation where appropriate
- External validation on an independent dataset
- Subgroup/error analysis
- Explainability analysis
- Comparison against additional architectures
- Manuscript-ready tables and figures

## References

- PyTorch Transfer Learning Tutorial:
  https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial
- TorchVision models:
  https://docs.pytorch.org/vision/main/models
- Grad-CAM:
  https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html
- Dataset:
  https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
