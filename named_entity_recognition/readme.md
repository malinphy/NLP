USAGE
---------------
REQUIREMENTS
---------------
```
python version: 3.7.14
numpy version: 1.21.6
pandas version: 1.3.5
regex version: 2.2.1
tensorflow version: 2.10.0
keras version: 2.10.0
tf_hub version: 0.12.0
tf_text version: 2.10.0
tokenizers version: 0.13.0
sklearn version: 1.0.2
seaborn version: 0.11.2
```
EVALUATION
---------------
###### Classification Report
```
F1 score: 0.917871611886156

precision    recall  f1-score   support

       B-LOC       0.69      0.72      0.71      2231
      B-MISC       0.70      0.62      0.66      1049
       B-ORG       0.78      0.64      0.70      1430
       B-PER       0.76      0.64      0.70      1973
       I-LOC       0.68      0.69      0.69       264
      I-MISC       0.74      0.41      0.52       367
       I-ORG       0.64      0.50      0.56       794
       I-PER       0.70      0.61      0.65      1358
           O       0.95      0.97      0.96     49527

    accuracy                           0.92     58993
   macro avg       0.74      0.64      0.68     58993
weighted avg       0.91      0.92      0.91     58993
```
             
###### Confusion Matrix For Evaluation Set

![image](https://user-images.githubusercontent.com/55249305/191540581-88ddb58d-bfd2-4b57-bb69-aa91c3fcc476.png)

Example
---------------
```
test_sentence : ['CRICKET - LEICESTERSHIRE TAKE OVER AT TOP AFTER INNINGS VICTORY .']
test_tags : O O B-ORG O O O O O O O O
pred_tags : O O B-ORG O O O O O O O O
```
