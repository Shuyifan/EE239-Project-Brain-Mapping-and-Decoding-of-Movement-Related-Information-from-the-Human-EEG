import conv_util
from load import load_data
from load import load_data_no_EOG
from load import load_data_with_fq_domain

# Run the shallow net
data = load_data_no_EOG()
conv_util.shallow_net(data)

# Run the neural net deeper comparing to the previous one
data = load_data_no_EOG()
conv_util.deep_net(data)

# Use the DFT to expand the feature from 22 channels to 44 channels, which containing 22 (23 - 44) channels in frequency domain
data = load_data_with_fq_domain()
conv_util.shallow_net_frequency(data)