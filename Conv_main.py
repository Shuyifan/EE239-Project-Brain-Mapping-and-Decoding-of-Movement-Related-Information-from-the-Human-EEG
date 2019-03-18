import conv_util
from load import load_data
from load import load_data_no_EOG
from load import load_data_with_fq_domain

#data = load_data()
#data = load_data_with_fq_domain()
data = load_data_no_EOG()
conv_util.deep_net(data)