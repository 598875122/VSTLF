# Very Short-Term Power Load Forecasting with Transformer and BiLSTM

This project focuses on Very Short-Term Load Forecasting (VSTLF) using a hybrid model combining Transformer architecture and Bidirectional Long Short-Term Memory (BiLSTM) networks. Leveraging a public dataset, we perform statistical analysis to identify key features influencing power load forecasting and train the hybrid model. Our approach surpasses several state-of-the-art methods in predictive performance. Additionally, we use Shapley Additive Explanations (SHAP) for feature importance analysis, enhancing the model's interpretability.




## Project Map
![img.png](pic/xmind.png)


## Requirements
* python>=3.8

* pytorch>=1.12.1

* numpy>=1.15.4

* pandas>=1.5.2

* shap==0.46.0

  

## Directory Structure

- data-source: Data source links
- data-process:Data processing, including normalization and dataset splitting
- **Transformer-LSTM**, **Transformer** directories: Corresponding model structures, hyperparameters, training strategies, and performance evaluation
- pth:Stores model weights after training



## Training

* STEP 1. Download the dataset and run data preprocessing.
```
python data_new.py
python data_person15.py  # Person-related correlation feature data
```
* STEP 2. Run the corresponding model for training and performance evaluation.
```
# example
python transformer_bilstm.py  #Includes interpretability analysis
python lstm.py
python GRU.py
python Lstm_EN-DE.py
python Transformer_lstm.py
python Transformer.py
python train_windowslip.py  #SimpleInformer
```
## Training Loss Curve
![img.png](pic/img.png)
## Prediction
![img_1.png](pic/img_1.png)

## Shap
![img_2.png](pic/img_2.png)

## Results
![image-3.png](pic/image_3.png)