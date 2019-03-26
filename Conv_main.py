import conv_util
from load import load_data

"--------------------------------------------------------------------------------------------------------------------------------------------------------"
"Run the shallow net" 

# Run the test on single person (from 0 to 8). In this example, we test person no. 8.
data = load_data(load_all=False, people=8, EOG=False, frequency=False)
conv_util.shallow_net(data)

# Run the test on the whole dataset.
data = load_data(load_all=True, people=None, EOG=False, frequency=False)
conv_util.shallow_net(data)

"--------------------------------------------------------------------------------------------------------------------------------------------------------"
"Run the deep net" 

# Run the neural net deeper comparing to the previous one
data = load_data(load_all=False, people=8, EOG=False, frequency=False)
conv_util.deep_net(data)

"--------------------------------------------------------------------------------------------------------------------------------------------------------"
"Use the DFT to expand the feature from 22 channels to 44 channels, which containing 22 (23 - 44) channels in frequency domain"

data = load_data(load_all=False, people=8, EOG=False, frequency=True)
conv_util.shallow_net_frequency(data)