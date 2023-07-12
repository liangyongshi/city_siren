import torch
import math
import numpy as np

# def initialize_fourier_mapping_vector(m,sigma):
#    mean=torch.tensor(np.zeros((m,3)))
#    std=torch.tensor(np.ones((m,3)))*sigma
#    B = np.random.normal((m, 3)) * sigma
#    return B

def fourier_mapping(coords,scale):
  """concate coords and fourier mapping
  """
  np.random.seed(1234)
  B = np.random.normal(size=(128, 3))*scale
  B = torch.from_numpy(B).cuda()
  sin_features = torch.sin((2 * np.pi * (torch.matmul(coords.float(), B.t().float()))))
  cos_features = torch.cos((2 * np.pi * (torch.matmul(coords.float(), B.t().float()))))
  features = torch.cat((sin_features, cos_features), axis=-1).cuda()
  co_features = torch.cat((coords.float(),features.float()),axis=-1).cuda()
  return co_features
