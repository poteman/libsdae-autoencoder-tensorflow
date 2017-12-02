from deepautoencoder import StackedAutoEncoder
import numpy as np


x = np.random.rand(100,5)
x2 = np.random.rand(100,5)

model = StackedAutoEncoder(dims=[5,6],activations=['relu','relu'],noise='gaussian',epoch=[10000,500],
                           loss='rmse',lr=0.007,batch_size=50,print_step=200)



# usage 1: encoding same data
result = model.fit_transform(x)
print(result.shape)

# usage 2
model.fit(x)
result2 = model.transform(x2)
print(result2)