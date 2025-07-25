from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Any, Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
import numpy as np
import random
import math
from sklearn.model_selection import KFold


import torch
import torch.nn as nn
import torch.nn.functional as F

import time


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args
    
    def forward(self, x):
        return x.view(self.shape)


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder: 224x224 -> 112x112 -> 56x56 -> 28x28 -> 14x14 -> 7x7
        self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, stride=2, kernel_size=3, bias=False, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout2d(0.1),
                #
                nn.Conv2d(32, 64, stride=2, kernel_size=3, bias=False, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout2d(0.1),
                #
                nn.Conv2d(64, 128, stride=2, kernel_size=3, bias=False, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout2d(0.1),
                #
                nn.Conv2d(128, 256, stride=2, kernel_size=3, bias=False, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout2d(0.1),
                #
                nn.Conv2d(256, 512, stride=2, kernel_size=3, bias=False, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout2d(0.1),
                #
                nn.Flatten(),
        )    
        
        # 512 * 7 * 7 = 25088
        self.z_mean = torch.nn.Linear(25088, 512)
        self.z_log_var = torch.nn.Linear(25088, 512)
        
        # Decoder: 7x7 -> 14x14 -> 28x28 -> 56x56 -> 112x112 -> 224x224
        self.decoder = nn.Sequential(
                torch.nn.Linear(512, 25088),
                Reshape(-1, 512, 7, 7),
                #
                nn.ConvTranspose2d(512, 256, stride=2, kernel_size=3, padding=1, output_padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout2d(0.1),
                #
                nn.ConvTranspose2d(256, 128, stride=2, kernel_size=3, padding=1, output_padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout2d(0.1),
                #
                nn.ConvTranspose2d(128, 64, stride=2, kernel_size=3, padding=1, output_padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout2d(0.1),
                #
                nn.ConvTranspose2d(64, 32, stride=2, kernel_size=3, padding=1, output_padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout2d(0.1),
                #
                nn.ConvTranspose2d(32, 3, stride=2, kernel_size=3, padding=1, output_padding=1),
                #
                nn.Sigmoid()
                )

    def encoding_fn(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return encoded

    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn_like(z_mu)
        z = z_mu + eps * torch.exp(z_log_var/2.) 
        return z
        
    def forward(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        decoded = self.decoder(encoded)
        return encoded, z_mean, z_log_var, decoded


def train_vae_v1(num_epochs, model, optimizer, device, 
                 train_loader, loss_fn=None,
                 logging_interval=100, 
                 skip_epoch_stats=False,
                 reconstruction_term_weight=1,
                 lr_scheduler=None,
                 save_model=None):
    
    log_dict = {'train_combined_loss_per_batch': [],
                'train_combined_loss_per_epoch': [],
                'train_reconstruction_loss_per_batch': [],
                'train_kl_loss_per_batch': []}

    if loss_fn is None:
        loss_fn = F.mse_loss

    start_time = time.time()
    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, _) in enumerate(train_loader):

            features = features.to(device)

            # FORWARD AND BACK PROP
            encoded, z_mean, z_log_var, decoded = model(features)
            
            # total loss = reconstruction loss + KL divergence
            #kl_divergence = (0.5 * (z_mean**2 + 
            #                        torch.exp(z_log_var) - z_log_var - 1)).sum()
            kl_div = -0.5 * torch.sum(1 + z_log_var 
                                      - z_mean**2 
                                      - torch.exp(z_log_var), 
                                      dim=1) # sum over latent dimension

            batchsize = kl_div.size(0)
            kl_div = kl_div.mean() # average over batch dimension
    
            pixelwise = loss_fn(decoded, features, reduction='none')
            pixelwise = pixelwise.view(batchsize, -1).sum(dim=1) # sum over pixels
            pixelwise = pixelwise.mean() # average over batch dimension
            
            loss = reconstruction_term_weight*pixelwise + kl_div * 0.1
            
            optimizer.zero_grad()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            # UPDATE MODEL PARAMETERS
            optimizer.step()
            
            # LOGGING
            log_dict['train_combined_loss_per_batch'].append(loss.item())
            log_dict['train_reconstruction_loss_per_batch'].append(pixelwise.item())
            log_dict['train_kl_loss_per_batch'].append(kl_div.item())
            
            if not batch_idx % logging_interval:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
                      % (epoch+1, num_epochs, batch_idx,
                          len(train_loader), loss))
        if not skip_epoch_stats:
            model.eval()
            
            with torch.set_grad_enabled(False):  # save memory during inference
                
                train_loss = compute_epoch_loss_autoencoder(
                    model, train_loader, loss_fn, device)
                print('***Epoch: %03d/%03d | Loss: %.3f' % (
                      epoch+1, num_epochs, train_loss))
                log_dict['train_combined_loss_per_epoch'].append(train_loss.item())

            if lr_scheduler is not None:
                lr_scheduler.step(train_loss)

        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    if save_model is not None:
        torch.save(model.state_dict(), save_model)
    
    return log_dict

def compute_epoch_loss_autoencoder(model, data_loader, loss_fn, device):
    model.eval()
    curr_loss, num_examples = 0., 0
    with torch.no_grad():
        for features, _ in data_loader:
            features = features.to(device)
            logits = model(features)[3]
            loss = loss_fn(logits, features, reduction='sum')
            num_examples += features.size(0)
            curr_loss += loss

        curr_loss = curr_loss / num_examples
        return curr_loss

class KDE(metaclass=ABCMeta):
    '''
    Abstract class for Kernel Density Estimation.
    '''
    def __init__(self, points: Tensor, bandwidth: float) -> None:
        '''
        Initialize the KDE class.

        Arguments:
            points:     Tensor, points to be used for the KDE
            bandwidth:  float, bandwidth of the KDE
        '''
        self.device = points.device
        self.dtype = points.dtype
        self.points = points
        self.bandwidth = bandwidth

    @abstractmethod
    def __call__(self, query: Tensor) -> Tensor:
        '''
        Evaluate the KDE at the query points.

        Arguments:
            query:      Tensor, query points
        Returns:
            Tensor, values of the KDE at the query points
        '''
        pass

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> None:
        '''
        Move all attributes that are tensors to the specified device and dtype.

        Arguments:
            device:     torch.device | str | None, device to move the attributes to
            dtype:      torch.dtype | None, dtype to move the attributes to
        '''
        assert device is None or isinstance(device, torch.device) or isinstance(device, str),\
            "device must be a torch.device, a string or None"
        assert dtype is None or isinstance(dtype, torch.dtype),\
            "dtype must be a torch.dtype or None"
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        for key in self.__dict__.keys():
            if torch.is_tensor(self.__dict__[key]):
                self.__dict__[key] = self.__dict__[key].to(device=device, dtype=dtype)
                

class GaussianKDE(KDE):
    '''
    Class for Gaussian Kernel Density Estimation.
    '''
    def __init__(self, points: Tensor, bandwidth: float) -> None:
        super().__init__(points, bandwidth)

    def __call__(self, query: Tensor) -> Tensor:
        norms = torch.sum((query[:, None, ...] - self.points[None, ...]) ** 2, dim=-1)
        return torch.logsumexp(-1. * norms / (2 * self.bandwidth ** 2), dim=-1)\
            - math.log(len(self.points) * self.bandwidth * math.sqrt(2 * math.pi))


def train_n_kdes(X: Tensor, Y: Tensor, n_folds: int = 5,
                 steps: int = 20, numclasses: int = 2) -> Tuple[GaussianKDE, ...]:
    '''
    Train n KDEs on the n-class dataset.

    Arguments:
        X:          Tensor, dataset
        Y:          Tensor, labels
        n_folds:    int, number of folds for cross-validation
        steps:      int, number of steps for the grid search
        numclasses: int, number of classes
    Returns:
        Tuple of GaussianKDE, KDEs for each class
    '''

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    best_sigmas = [0 for _ in range(numclasses)]
    
    for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
        print(f"\rTraining fold {fold}/{n_folds}", end="")
        
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = Y[train_index], Y[val_index]

        # Find best sigma for each class
        for i in range(numclasses):
            sigma = train_kde(X_train[y_train == i], X_val[y_val == i], steps)
            best_sigmas[i] += sigma

    # Average the best sigmas across folds
    best_sigmas = [sigma / n_folds for sigma in best_sigmas]

    print('')
    print(f'Final sigma for each class = {best_sigmas}')

    # Create final KDEs using all points from each class
    kdes = [GaussianKDE(X[Y == i], best_sigmas[i]) for i in range(numclasses)]

    return kdes

def train_kde(X_train: Tensor, X_val: Tensor, steps: int = 50) -> float:
    sigmas = [0.01, 1, 5, 10, 20, 50, 100, 200, 500]
    best_sigma = grid_search(X_train, X_val, sigmas)
    
    # Fine-tune sigma
    sigma_range = [best_sigma / 2, best_sigma * 2]
    fine_sigmas = torch.linspace(sigma_range[0], sigma_range[1], steps).to(X_train.device)
    best_sigma = grid_search(X_train, X_val, fine_sigmas)

    return best_sigma

def grid_search(X_train: Tensor, X_val: Tensor, sigmas: List[float]) -> float:
    results = []
    for sigma in sigmas:
        kde = GaussianKDE(X_train, sigma)
        results.append(kde(X_val).mean().item())
    return sigmas[results.index(max(results))]



class GaussianMixture():

    def __init__(self, K: int, d: int, device: torch.device | str | None = None,
                 dtype: torch.dtype | None = None) -> None:
        '''
        Initialize the GaussianMixture class.

        Arguments:
            K:          int, initial number of components (will be updated during training)
            d:          int, dimension of the data
            device:     torch.device | str | None, device to move the attributes to
            dtype:      torch.dtype | None, dtype to move the attributes to
        '''
        self.K = K
        self.d = d
        self.device = device
        self.dtype = dtype
        self.loglikelihood = -torch.inf
        self.mu = None
        self.var = None
        self.pi = None


    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> None:
        '''
        Move all attributes that are tensors to the specified device and dtype.

        Arguments:
            device:     torch.device | str | None, device to move the attributes to
            dtype:      torch.dtype | None, dtype to move the attributes to
        '''
        assert device is None or isinstance(device, torch.device) or isinstance(device, str),\
            "device must be a torch.device, a string or None"
        assert dtype is None or isinstance(dtype, torch.dtype),\
            "dtype must be a torch.dtype or None"
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        for key in self.__dict__.keys():
            if torch.is_tensor(self.__dict__[key]):
                self.__dict__[key] = self.__dict__[key].to(device, dtype)


    def fit(self, x, iters=100, eps=1e-9, init=True):
        '''
        Fit the GMM to the data.

        Arguments:
            x:          Tensor, data
            iters:      int, number of iterations
            eps:        float, convergence criterion
            init:       bool, whether to initialize the parameters
        '''
        with torch.no_grad():
            if init:
                self.mu.data = self.K_means(x)
                self.var = torch.ones(1, self.K, self.d, dtype=self.dtype, device=self.device) * 5
                self.pi = torch.ones(1, self.K, 1, dtype=self.dtype, device=self.device) / self.K

            for i in range(iters):
                ll = self.loglikelihood

                # E step
                weighted_ll = self.get_log_prob(x) + torch.log(self.pi)
                log_belief = weighted_ll - torch.logsumexp(weighted_ll, dim=1, keepdim=True)

                # M step
                belief = torch.exp(log_belief)
                pi = self.numstab(torch.sum(belief, dim=0, keepdim=True))
                self.pi.data = pi / x.size(0)
                self.mu.data = torch.sum(belief * x[:, None, :], dim=0, keepdim=True) / pi
                x_ = torch.sum(belief * x[:, None, :] ** 2, dim=0, keepdim=True)
                mu_ = torch.sum(belief * x[:, None, :] * self.mu, dim=0, keepdim=True)
                self.var.data = (x_ - 2 * mu_) / pi + self.mu ** 2

                # if covariance gets close to singular, choose new corresponding
                # mean randomly and reset covariance
                if torch.any(sing := (self.var <= 1e-6)):
                    for j, s in enumerate(sing[0]):
                        if torch.any(s):
                            self.mu[0][j] = x[random.randint(0, len(x) - 1)]
                            self.var[0][j] = torch.ones_like(self.var[0][j]) * 2

                # update loglikelihood and check criterion
                self.loglikelihood = self(x).mean()
                if self.loglikelihood - ll < eps:
                    break
            return


    def get_log_prob(self, x):
        '''
        Get the log probability of the data under the GMM.

        Arguments:
            x:          Tensor, data
        Returns:
            Tensor, log probability of the data
        '''
        mu = self.mu
        var = self.var

        log_det = torch.sum(torch.log(var), dim=2, keepdim=True) #torch.log(var)
        log_p = torch.sum((mu - x[:, None, :]) ** 2 / var, dim=2, keepdim=True)

        return -.5 * (self.d * np.log(2 * torch.pi) + log_det + log_p)


    def __call__(self, x):
        '''
        Evaluate the GMM at the data points.

        Arguments:
            x:          Tensor, data
        Returns:
            Tensor, log probability of the data
        '''
        weighted_ll = self.get_log_prob(x) + torch.log(self.pi)
        return torch.logsumexp(weighted_ll, dim=1).squeeze()


    def numstab(self, a):
        b = torch.ones_like(a) * 1e-6
        return torch.where(abs(a) < b, b, a)


    def K_means(self, x, iters=50):
        '''
        K-means clustering to find means of the components. Checks
        10 values for the number of centroids between 0.5K and 1.5K.
        Sets self.K to the best K and returns the centroids.

        Arguments:
            x:          Tensor, data
            iters:      int, number of iterations
        Returns:
            Tensor, means of the components
        '''
        K1 = math.ceil(self.K * 0.5)
        K2 = math.ceil(self.K * 1.5)
        Ks = torch.unique(torch.linspace(K1, K2, 10).round().int()).to(self.device)
        N = x.size(0)
        d = self.d

        best_K = None
        best_centroids = None
        best_score = float('inf')

        for K in Ks:
            print(f'\rChecking K = {K}, {K1} <= K <= {K2}', end='', flush=True)
            # initialize the centroids as K random points in the data set
            indices = torch.randperm(N, device=self.device)[:K]
            centroids = x[indices, :].clone()

            for i in range(iters):
                # compute square distance of all points to all centroids and find
                # the closest centroid for every point
                dists = ((x.view(N, 1, d) - centroids.view(1, K, d)) ** 2).sum(-1)
                closest = dists.argmin(dim=1).view(-1)

                for k in range(K):
                    # update the centroids to be the mean of the points for which
                    # the centroid was the closest one
                    points = x[closest == k]
                    if len(points):
                        centroids[k] = points.mean(0)

            # Evaluate the clustering and save the best one
            score = dists.min(axis=1).mean()
            if score < best_score:
                best_score = score
                best_K = K
                best_centroids = centroids

        print('')
        self.K = best_K
        return best_centroids.view(1, K, d)
    

def train_n_GMMs(X: Tensor, Y: Tensor, _: Any = None,
                 K: int = 5, numclasses: int = 2) -> Tuple[GaussianMixture, ...]:
    '''
    Train n GMMs on the n-class dataset.

    Arguments:
        X:          Tensor, data
        Y:          Tensor, labels
        K:          int, initial number of components
        numclasses: int, number of classes
    Returns:
        Tuple of GaussianMixture, GMMs for each class
    '''
    GMMs = []
    for i in range(numclasses):
        print(f"Training GMM for class {i+1}/{numclasses}")
        GMMs.append(GaussianMixture(K, X.shape[-1], device=X.device, dtype=X.dtype))
        GMMs[i].fit(X[Y == i])
    return GMMs


def evaluate_DEs(des, x, y):
    '''
    Evaluate the density estimators at the data points.

    Arguments:
        des:        Tuple of KDE, density estimators
        x:          Tensor, data
        y:          Tensor, labels
    Returns:
        Tensor, mean log probability of the data
    '''
    kdeloss = torch.empty((x.size(0),), device=x.device, dtype=x.dtype)
    mapidx = [i for i in range(len(des))] if len(des) != 2 else [-1, 1]
    for i in range(len(des)):
        mask = (y == mapidx[i]).view(-1)
        if mask.any():
            kdeloss[mask] = des[i](x[mask])

    return kdeloss.mean()


class LinearModel:
    def __init__(self, w: Tensor, b: Tensor):
        self.w = w
        self.b = b

    def to(self, device, dtype):
        self.w = self.w.to(device=device, dtype=dtype)
        self.b = self.b.to(device=device, dtype=dtype)

    def parameters(self):
        return iter([self.w, self.b])

    def __call__(self, x):
        return x @ self.w.T + self.b
    

class MNISTCNN(nn.Module):

    def __init__(self, shape=(1, 28, 28)):
        super(MNISTCNN, self).__init__()

        self.shape = shape
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout(0.1)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)

    def forward(self, x):
        x = x.view(-1, *self.shape)
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out
    

def train(classifier, dataloader, epochs=10, lr=1e-3, loss_fn=nn.CrossEntropyLoss(), trf: Callable = lambda x: x):

    device = next(classifier.parameters()).device
    classifier.train()
    class_opt = optim.Adam(classifier.parameters(), lr=lr, betas=(0.5, 0.999))
    sm = nn.Softmax(dim=1)

    for epoch in range(epochs):
        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            classout = sm(classifier(trf(x)))
            classloss = loss_fn(classout, y)
            class_opt.zero_grad()
            classloss.backward()
            class_opt.step()

            print(f'\rEpoch: {epoch+1}/{epochs}, Batch: {i+1}/{len(dataloader)}', end='')
    print('')