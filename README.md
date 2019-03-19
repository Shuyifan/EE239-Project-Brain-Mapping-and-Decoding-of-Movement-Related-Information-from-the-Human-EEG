# EE239-Project

## CNN Part

### How to run
```
python Conv_main.py
```
To run different different model, uncomment the corresoponding part in the Conv_main.py.

### Python Files for this Part
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
* load_data(): load the data (including EOG + EEG)
* load_data_no_EOG(): load the data (without EOG channels)
* load_data_with_fq_domain(): load the data and use DFT to extend from 22 channels to 44 channels, where 1 - 22 channels are original signals in the time domain, and 23 - 44 channels are the corresponding signals in the frequency domain.

## RNN Part
### How to run
Just run the jupyter notebook, "RNN.ipynb". "RNN.ipynb" is the keras based implementation of the LSTM networks. It includes both simple RNN and stacked BRNN models and their experiments on data sets with different preprocessing method.
