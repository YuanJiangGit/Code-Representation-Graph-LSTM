# Code Representation

## Downstream tasks of code representation

**Program classification** aims to classify newly added source
files into different categories according to their functionalities
in the software development process

## Dataset

The dataset we use is OJ datasets, including OJ-Data-1, OJ-Data-2 and OJ-All.

### Data Format

1. resources/dataset/OJ-Data-*/programs.pkl is stored in pickle format. Each row in this file represents one function and its label.  One row is illustrated below.

   - **id:** index of the example
      
   - **Code:** the code fragment

   - **label** index of the example


### Data Statistics

Data statistics of the dataset are shown in the below table:

|       | #Examples | #Program Tasks |
| ----- | :-------: |:-------: |
| OJ-Data-1 |  52,000  | 104 |
| OJ-Data-2    |  52,000  | 104 |
| OJ-All   |  104,000  | 208 |

You can get data using the following command.

```python
import os
import pandas as pd
data_path = '../resources/dataset/OJ-All/programs.pkl'
if os.path.exists(data_path):
    data = pd.read_pickle(data_path)
```

## Pipeline-Prepare input for our grah-lstm model

We also provide a pipeline that generates inputs for our model on this task. 
### Dependency

- gensim
- networkx
- dgl
- nltk
- numpy
- pandas
- scikit_learn
- torch

### Tree-sitter

If the built file "parser/my-languages.so" doesn't work for you, please rebuild as the following command:

```shell
cd parser1
bash build.sh
cd ..
```

```shell
python DataProcess/Pipline.py
```


## Training and Evaluation Entry

We provide a script to train and evaluate our model for this task, and report Accuracy score

```shell
python Entry/train_graph_lstm.py
```

### Result

```bash
OJ-All
[Epoch: 100/100] Train Loss: 0.0526, Val Loss: 0.0540, Train result: 0.9773022361144506, Test result: 0.9660623692625808
```

