import conv_util
from load import load_data
from load import load_data_no_EOG

#data = load_data()
data = load_data_no_EOG()
conv_util.deep_net(data)