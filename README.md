# EE239 Project: Deep Learning for Brain Mapping and Decoding of Movement-Related Information from the Human EEG
## Contruibutors
Yifan Shu: evanshu@g.ucla.edu <br />
Ruoyu Peng: jimmypry@g.ucla.edu <br/>
Haiqi Xiao: haiqixiao@ucla.edu <br/>
Yifeng Zhang: tjyifengzhang@outlook.com <br />
## CNN Part

### How to run
```
python Conv_main.py
```
To run different different model, uncomment the corresoponding part in the conv_main.py.

For this project, we used the GPU version of the Tensorflow to boost the running speed. Our environment can be shown in detail in Environment.txt.

### Python Files for the Part
##### Conv_main.py
The main program for the code.
##### conv_util.py
The utility function for the convolutional network. Contains the following function:
* get_batch_id(batchsize, datalen): get the indexs for a batch given the length of the whole dataset.
* shallow_net(data_list): use the shallow net to train and test the model.
* deep_net(data_list): use the deep net to train and test the model.
* shallow_net_frequency(data_list): use the shallow net to train and test the model using the dataset with frequency domain channle.
##### load.py
Contains the function to load the data. Includes the following function:
* one_hot(Y): do the one-hot coding for the target.
* load_data(): load the data. By feeding diffrerent parameter, we can load the data in different format, further details can be seen in the comment in the load.py.

## RNN Part
### How to run
Just run the jupyter notebook, "RNN.ipynb". "RNN.ipynb" is the Keras based implementation of the LSTM networks. It includes both simple RNN and stacked BRNN models and their experiments on data sets with different preprocessing method.