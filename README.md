## Arabidopsis expression prediction

### Step1:

Expression data download and processing, follow the steps in notebooks  [data_download_mergeRep.ipynb](https://github.com/zzh24zzh/deepPlant/blob/main/data_download_mergeRep.ipynb) and 
[processing.ipynb](https://github.com/zzh24zzh/deepPlant/blob/main/processing.ipynb)


### Step2:

Preparing training and validation set

```
python split_ara_data.py
```

### Step3:
Traing the model
```
bash run.sh
```



