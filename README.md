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

## 2. Experiment

### Dataset

- [Korean HateSpeech Dataset](https://github.com/kocohub/korean-hate-speech)



자세한 실험 정보는 해당 [링크](https://jet-rook-fae.notion.site/35b7edef4a3c4c9780f8e8e27bbc1bb8)를 참조해주세요.

## 3. Final Model



## 4. Flow Chart

### System Architecture

![image](https://user-images.githubusercontent.com/48538655/146737900-f9885822-08d7-4427-b1b1-3c2c9330ad78.png)

### Pipeline

![image](https://user-images.githubusercontent.com/48538655/146738181-85996171-e84f-451a-85ca-165098608523.png)

## 5. How to Use  (추후 수정)

### Dependencies

- pandas == 

### Install Requirements

```
pip install -r requirements.txt
```

### Project Tree

```
|-- README.md
|-- base
|   |-- __init__.py
|   |-- base_data_loader.py
|   |-- base_model.py
|   `-- base_trainer.py
|-- config.json
|-- data_loader
|   |-- __pycache__
|   |   `-- data_loaders.cpython-37.pyc
|   `-- data_loaders.py
|-- logger
|   |-- __init__.py
|   |-- logger.py
|   `-- logger_config.json
|-- model
|   |-- loss.py
|   |-- metric.py
|   `-- model.py
|-- parse_config.py
|-- requirements.txt
|-- simple_test.py
|-- test.py
|-- train.py
|-- trainer
|   |-- __init__.py
|   `-- trainer.py
`-- utils
    |-- __init__.py
    |-- api_response.py
    |-- error_handler.py
    `-- util.py
```

### Getting Started

아래 명령어로 실행 가능합니다.

```
python ___.py
```

## 6. Reference

