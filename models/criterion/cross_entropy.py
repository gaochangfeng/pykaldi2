import torch

class SeqCrossEntorpy(torch.nn.CrossEntropyLoss):
    '''
        x:[B,T,D],y:[B,T,1]
    '''
    def forward(self,x,y):
        #super(self,x.view(-1, x.shape[2]), y.view(-1))
        x = x.contiguous()
        y = y.contiguous()
        return super(SeqCrossEntorpy, self).forward(x.view(-1, x.shape[2]), y.view(-1))