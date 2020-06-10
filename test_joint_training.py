import unittest
import torch 
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image
from torch.distributions.normal import Normal
from models.vae import VAE

from utils.misc import save_checkpoint
from utils.misc import LATENT_SIZE, IMAGE_RESIZE_DIM
import numpy as np

class TestJointTrain(unittest.TestCase):

    def test_vae_loss(self):