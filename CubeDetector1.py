import torch.nn as nn

class CubeDetector1(nn.Module):
  def __init__(self, in_dim, nkernels1=32, nkernels2=12, pool1=4, pool2=4):
    super(CubeDetector1, self).__init__()
    self.network = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=nkernels1, kernel_size=11, stride=1, padding=2),
      nn.BatchNorm2d(nkernels1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=pool1),
      nn.Conv2d(in_channels=nkernels1, out_channels=nkernels2, kernel_size=11, stride=1, padding=2),
      nn.BatchNorm2d(nkernels2),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=pool2),
      nn.Flatten(),
      nn.Linear((int(in_dim[0]*in_dim[1]/(pool1**2*pool2**2))*nkernels2)-696, 2)
    )
      
  def forward(self,input):
    return self.network(input)
