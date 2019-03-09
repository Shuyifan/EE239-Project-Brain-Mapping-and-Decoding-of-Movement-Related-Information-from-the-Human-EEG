import conv_util
from load import load_data

data = load_data()
conv_util.ConvNet_nocrop(data)