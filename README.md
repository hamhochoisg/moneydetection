# Vietnam Banknotes Classification - Using Machine Learning

## Project Motivation
The original idea from a kind guy wanted to make a banknotes recognition device to identify banknotes supporting the blind. The device will use an AI camera combined with a pre-trained machine learning model to recognize the banknotes.

## File Structure
- Banknotes_Classifier_Model.ipynb : This is the jupyter notebook file used to train the model. Using Transfer-Learning method from very good models from ImageNet competition
- Main.py : This is the python code file used to create the streamlit application for the project

## Project Process:
- **Step 1**: A lot of banknote images are collected to create a data library for the project, the banknotes are classified by value.
![1](https://user-images.githubusercontent.com/88182498/144193291-11860176-d1e0-4f85-90af-9129a0a7035c.png)

- **Step 2**: The data is cleaned again and preprocessed to increase model diversity
![image](https://user-images.githubusercontent.com/88182498/144193369-f09648f7-4c46-4d30-be33-df7cfb1aeb68.png)

- **Step 3**: Build a model and use Transfer Learning to inherit previous machine learning models.
![image](https://user-images.githubusercontent.com/88182498/144193515-324e8a87-921a-4424-9c37-96c497f197d5.png)

- **Step 4**: Trainning and Testing Model For Best Results
![image](https://user-images.githubusercontent.com/88182498/144193642-56d4743a-9b10-4d2f-a5a1-34f44c1fa606.png)

- **Step 5**: Real Test with Hidden Test Set
![image](https://user-images.githubusercontent.com/88182498/144193734-b2c28da1-faed-4f8e-a7f7-5dba5793a4f2.png)
![image](https://user-images.githubusercontent.com/88182498/144193800-4c0457fc-9126-4815-896f-1544ca69a11e.png)

## Conclude:
- The model gives very good results on training data but on hidden tests, the results are not good. Especially when predicting 500,000 banknote, money loses its angle or gets blurred
- The model does not analyze non-money images
