import torch
import numpy as np 
import numpy.ma as npm

from models.net.lstm import LSTMStack

def random_mask(data,p,fill_data=0):
    sh = np.shape(data)
    mask = np.random.rand(*sh) < p
    data[mask] = 0
    return data,mask

def main():
    print("hello")
    a = np.random.rand(2,4,3)
    ma,mask = random_mask(a,0.5)
    print(ma)
    ma = torch.from_numpy(ma).float()
    pre_model = LSTMStack(3,256,3,0.1,True)
    out_layer = torch.nn.Linear(2*256,3,True)
    out,_ = pre_model(ma)
    out = out_layer(out)
    mask = torch.from_numpy(~mask).byte()
    loss = (ma - out).masked_fill(mask,0)
    loss = torch.norm(loss)
    print(loss)








if __name__ == "__main__":
    main()