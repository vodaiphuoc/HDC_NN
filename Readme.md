A method for retraining HDC model \
Overview: \
The proposed solution works mainly on binding (for bipolar) or XOR (for binary) \
which tends to decrease #-1 bits from high to low for positive sample and increase\
#1 bits from low to high for negative sample \

As a result, the constructed mask is used for flipping bit where -1 is flip and 1 is \
not to flip. The retrained model archieved 2% accuracy gain on Mnist dataset.