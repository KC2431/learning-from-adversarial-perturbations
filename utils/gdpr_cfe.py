import torch
from .Models import *
from .utils import freeze

class GDPR_CFE:
    def __init__(self, 
                 model,
                  max_image_range = 1.0,
                  min_image_range = 0.0, 
                  optimizer = torch.optim.Adam, 
                  iters: int=100, 
                  lamb: float=1.0,
                  lamb_cf: float=1e-2,
                  mode: str = 'natural',
                  device: str = 'cuda:0',
                  ):
        
        self.model = model
        self.max_image_range = max_image_range
        self.min_image_range = min_image_range
        self.optimizer = optimizer
        self.iters = iters
        self.device = device
        self.loss_lambda = lamb
        self.lambda_cf = lamb_cf
        self.mode=mode

    def loss(self, logits, y_true, x, x_adv,mode, lamb):
        assert y_true.shape == (x.shape[0],), "Target label is expected to be a tensor of shape (n,)"
        assert len(logits.shape) == 2 if mode in ['natural','natural_binary'] else len(logits.shape) == 1, "Logits is expected to be a tensor of shape (n, num_classes)"

        if mode == 'natural':
            return torch.nn.CrossEntropyLoss()(logits, y_true) + lamb * torch.sum((x - x_adv).pow(2), dim = (1,2,3)).mean()
        elif mode == 'natural_binary':
            return torch.nn.CrossEntropyLoss()(logits, y_true) + lamb * torch.sum((x - x_adv).pow(2), dim = (1,2,3)).mean()
        elif mode == 'artificial':
            return (-logits * y_true.cuda()).exp().mean() + lamb * torch.sum((x - x_adv).pow(2), dim = 1).mean()
        else:
            NotImplementedError
    def get_perturbations(self, x, y):

        assert x.min() >= self.min_image_range and x.max() <= self.max_image_range, f"Data is expected to be in the specified range [{self.min_image_range}, {self.max_image_range}]"        
        assert x.shape[0] == y.shape[0], "Data and target label are expected to have the same number of samples"

        print(f'Using GDPR CFE')

        freeze(self.model)
        self.model.eval()

        lamb = self.loss_lambda

        x_adv = x + 0.01 * torch.rand_like(x)
        x_adv = torch.clamp(x_adv, self.min_image_range, self.max_image_range)
        optim = self.optimizer([x_adv],lr=self.lambda_cf)

        for i in range(self.iters):
            optim.zero_grad()
            x_adv.requires_grad = True
            output = self.model(x_adv)
            loss = self.loss(logits=output, 
                                y_true=y.squeeze(1), 
                                x=x, 
                                x_adv=x_adv,
                                mode=self.mode,
                                lamb=lamb
                                )
            loss.backward()
            optim.step()
            
            x_adv.data = torch.clamp(x_adv.data, self.min_image_range, self.max_image_range)
            
            if i % 10 == 0:
                print("Iter: {}, Loss: {}".format(i, loss.item()))
                #lamb *= 2

        with torch.no_grad():
            if self.mode == 'natural':
                print(self.model(x_adv).argmax(1).eq(y.squeeze(1)).float().mean().item())
            
            

        assert x_adv.min() >= self.min_image_range and x_adv.max() <= self.max_image_range, "Adversarial image needs to be in image range."
        return x_adv.detach()

if __name__ == '__main__':
    print("Hello")
