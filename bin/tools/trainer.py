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

    def train_batch(self,data,label,*arg,max_grad_norm=5,**args):
        out = self.forward(data,*arg,**args)
        loss = self.loss_fn(out,label,*arg,**args)
        self._backward(loss,max_grad_norm)
        return loss

    def _backward(self,loss,max_grad_norm):
        '''
            you can override this method to realize different trainer
        '''
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient Clipping
        if max_grad_norm != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        self.optimizer.step()

    def update_opt(self,func,*arg,**args):
        if func is None:
            return
        else:
            #应该可以用@函数实现，func其他参数的确认
            self.optimizer = func(self.optimizer,*arg,**args)


def update_opt_func(fn):
    '''
        优化器更新函数的修饰器，update_opt只能接受被该函数修饰的函数
        被修饰的函数一定要返回一个优化器的类
    '''
    def updata(optimizer,*args,**kwargs):
        if not isinstance(optimizer,torch.optim.Optimizer):
            raise TypeError("update_opt only accept torch.optim.Optimizer as the first parameter")
        return fn(optimizer,*args,**kwargs)
    return updata