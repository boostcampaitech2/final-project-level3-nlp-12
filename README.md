# Malicious Comments Collection System

## 1. Introduction

![image](https://user-images.githubusercontent.com/48538655/146724216-66d3da83-5024-4253-b170-1acd4449a344.png)

  인터넷이 발달하면서 특정 인물들에 대한 무분별한 악플들이 사람들을 괴롭히고 있습니다. 이런 악플러를 신고 및 고소를 하는데 증거 수집은 필수이지만 오랜 시간을 들여 증거수집이 필요합니다. 특히, 현재 프로세스는 회사나 개인 차원에서 직접 수집을 하거나 팬들의 제보를 통해 이루어지므로 비효율적이며 수동적입니다. 따라서 이런 점을 개선하고자 해당 프로젝트를 진행하게 되었습니다.

  **Malicious Comments Collection System**는 악플을 수집하고 악플을 검토하는 부분을 자동화하는데에 목적이 있습니다. 수집된 자료들은 추후 고소 목적으로 활용이 될 것입니다. 

### Team AI-it

> "아-잇" 이라고 발음되는 것이 키치하게 재밌어서 팀명으로 정해보았습니다.

#### Members

|                            이연걸                            |                            김재현                            |                            박진영                            |                            조범준                            |                            진혜원                            |                            안성민                            |                            양재욱                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src='https://avatars.githubusercontent.com/u/48538655?v=4' height=80 width=80px></img> | <img src='https://avatars.githubusercontent.com/u/83448285?v=4' height=80 width=80px></img> | <img src='https://avatars.githubusercontent.com/u/34739974?v=4' height=80 width=80px></img> | <img src='https://avatars.githubusercontent.com/u/20266073?v=4' height=80 width=80px></img> | <img src='https://avatars.githubusercontent.com/u/39722108?v=4' height=80 width=80px></img> | <img src='https://avatars.githubusercontent.com/u/81609329?v=4' height=80 width=80px></img> | <img src='https://avatars.githubusercontent.com/u/56633607?v=4' height=80 width=80px></img> |
| [![Git Badge](http://img.shields.io/badge/-Github-black?style=flat-square&logo=github)](https://github.com/LeeYeonGeol) | [![Git Badge](http://img.shields.io/badge/-Github-black?style=flat-square&logo=github)](https://github.com/CozyKim) | [![Git Badge](http://img.shields.io/badge/-Github-black?style=flat-square&logo=github)](https://github.com/nazzang49) | [![Git Badge](http://img.shields.io/badge/-Github-black?style=flat-square&logo=github)](https://github.com/goattier) | [![Git Badge](http://img.shields.io/badge/-Github-black?style=flat-square&logo=github)](https://github.com/hyewon11) | [![Git Badge](http://img.shields.io/badge/-Github-black?style=flat-square&logo=github)](https://github.com/tttangmin) | [![Git Badge](http://img.shields.io/badge/-Github-black?style=flat-square&logo=github)](https://github.com/didwodnr123) |

### Contribution

- [`이연걸`](https://github.com/LeeYeonGeol) &nbsp; Project Management • Service Dataset • Front-end & Back-end Update • EDA
- [`김재현`](https://github.com/CozyKim) &nbsp; Modeling • Model Optimization • AutoML • EDA
- [`박진영`](https://github.com/nazzang49) &nbsp; Model Optimization • Application Cloud Release (GKE) • Service Architecture
- [`조범준`](https://github.com/goattier) &nbsp; Baseline Code • Modeling • Model Optimization • EDA
- [`진혜원`](https://github.com/hyewon11) &nbsp; Service Dataset • EDA • Front-end & Back-end Update
- [`안성민`](https://github.com/tttangmin)  &nbsp;EDA • Modeling
- [`양재욱`](https://github.com/didwodnr123) &nbsp; Front-end (Streamlit) • Back-end (FastAPI) • MongoDB •  EDA

## 2. Model

### Transformer + CNN & RNN based model (Best LB f1-score: 64.856)
![image](https://user-images.githubusercontent.com/20266073/147147702-ff94e551-ea1c-4b4e-bdd5-622a31680442.png)

### Clustering + KNN (Best LB f1-score: 66.192)
![image](https://user-images.githubusercontent.com/20266073/147147922-aebcf049-1f3f-49b3-954f-a9322d4ec901.png)

### 2nd / 67team (21.12.23 기준)
![image](https://user-images.githubusercontent.com/20266073/147148111-587f6ca2-0252-4237-ab63-bc9e919c3064.png)


## 3. Flow Chart

### System Architecture

![image](https://user-images.githubusercontent.com/39722108/147093220-ef42c25c-c240-4911-93c9-ec0eb81af432.png)

### Pipeline

![image](https://user-images.githubusercontent.com/48538655/146738181-85996171-e84f-451a-85ca-165098608523.png)

## 4. How to Use

### Install Requirements

```bash
pip install -r requirements.txt
```

### Project Tree

```
|-- automl
|-- base
|   |-- __init__.py
|   |-- base_data_loader.py
|   |-- base_model.py
|   └-- base_trainer.py
|-- data_loader
|   └-- data_loaders.py
|-- logger
|   |-- __init__.py
|   |-- logger.py
|   └-- logger_config.json
|-- model
|   |-- loss.py
|   |-- lr_scheduler.py
|   |-- metric.py
|   └-- model.py
|-- prototype
|-- tokenizer
|   |-- special_tokens_map.json
|   |-- tokenizer_config.json
|   └-- vocab.txt
|-- trainer
|   |-- __init__.py
|   └-- trainer.py
|-- config.json
|-- config_automl_test.json
|-- parse_config.py
|-- pkm_config.json
|-- requirements.txt
|-- simple_test.py
|-- test.py
|-- test_automl.py
|-- train.py
└-- utils
    |-- __init__.py
    |-- api_response.py
    |-- error_handler.py
    |-- memory.py
    |-- query.py
    |-- util.py
    └-- utils.py
```

### Getting Started
- Train & Validation
```python
python train.py -c config.json
```
- Inference
```python
python test.py -c config.json    # test_config.json
```

## 5. Demo (TODO)

## 6. Reference
- [Korean HateSpeech Detection Kaggle Competition](https://www.kaggle.com/c/korean-hate-speech-detection/data)
- [Korean HateSpeech Dataset](https://github.com/kocohub/korean-hate-speech)
- [BEEP! Korean Corpus of Online News Comments for Toxic Speech Detection](https://aclanthology.org/2020.socialnlp-1.4/)
