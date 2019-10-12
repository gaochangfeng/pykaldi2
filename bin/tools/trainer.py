import torch


class Trainer(torch.nn.Module):
    def __init__(self,model,criterion,optimizer):
        super(Trainer, self).__init__()
        self.model = model
        self.loss_fn = criterion
        self.optimizer = optimizer

    def forward(self,data,*arg,**args):
        self.model.train()
        return self.model(data,*arg,**args)

    def train_batch(self,data,label,*arg,**args):
        out = self.forward(data,*arg,**args)
        loss = self.loss_fn(out,label,*arg,**args)
        self._backward(loss)
        return loss

    def _backward(self,loss):
        '''
            you can override this method to realize different trainer
        '''
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient Clipping
        #norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        self.optimizer.step()
        self._update_opt()

    def _update_opt(self,func=None):
        if func is None:
            return
        else:
            #应该可以用@函数实现，func其他参数的确认
            self.optimizer = func(self.optimizer)