
import torch
import math


def pad(T, padding):
  torch_pad = torch.nn.ZeroPad2d((padding[1], padding[1], padding[0], padding[0]))
  return torch_pad(T)

def cross_correlation(F, G, padding = (0,0), stride = (1,1)):
  F = pad(F, padding)
  corr = torch.zeros((math.floor((F.shape[0] - G.shape[0])/stride[0]) +1, math.floor((F.shape[1] - G.shape[1])/stride[1]) + 1))
  i = 0
  for u in range(0,F.shape[0],stride[0]):
    j = 0
    for v in range(0,F.shape[1],stride[1]):
      if(u+G.shape[0] <= F.shape[0] and v+G.shape[1] <= F.shape[1]):
        sub_F = F[u:u+G.shape[0], v:v+G.shape[1]]
        corr[i,j] = (G * sub_F).sum()
      j += 1
    i += 1
  return corr


def pooling(F, pool_size, pool_type='max', padding = (0, 0), stride = (1, 1)):
  pad = torch.nn.ZeroPad2d((padding[1], padding[1], padding[0], padding[0]))
  F = pad(F)
  pooled = torch.zeros((math.floor((F.shape[0] - pool_size[0])/stride[0]) +1, math.floor((F.shape[1] - pool_size[1])/stride[1]) + 1))
  i = 0
  for u in range(0,F.shape[0],stride[0]):
    j = 0
    for v in range(0,F.shape[1],stride[1]):
      if(u+pool_size[0] <= F.shape[0] and v+pool_size[1] <= F.shape[1]):
        pool = F[u:u+pool_size[0], v:v+pool_size[1]]
        if(pool_type == 'max'):
          pooled[i,j] = torch.max(torch.flatten(pool))
        elif(pool_type == 'avg'):
          pooled[i,j] = torch.mean(torch.flatten(pool))
      j += 1
    i += 1
  return pooled

if __name__ == '__main__':
  A = torch.tensor([[0.0, 1.0, 2.0], 
                    [3.0, 4.0, 5.0], 
                    [6.0, 7.0, 8.0]])
  B = torch.tensor([[-1.0, 1.0], [2.0, 0.5]])
  print(cross_correlation(A,B, padding = (1,1), stride=(3,2)))
  
  print(pooling(A, (3, 3), 'max', padding = (1, 1), stride=(2, 2)))
  print(pooling(A, (3, 3), 'avg', padding = (1, 1), stride=(2, 2)))
