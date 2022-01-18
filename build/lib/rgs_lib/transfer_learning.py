class ResNet50Extractor(nn.Module):

  def __init__(self, output_size):
    super().__init__()
    self.resnet50 = resnet50(pretrained=True)
    self.freeze()
    self.resnet50.fc = nn.Sequential(
        nn.Linear(self.resnet50.fc.in_features, output_size),
    ) 
  
  def forward(self, x):
    return self.resnet50(x)

  def freeze(self):

    """
      A method to freeze gradient descent, so that the model weight does not change. 
      The goal is to adapt the model, or build a mini neural network on the model
    """
    for param in self.resnet50.parameters():
      param.requires_grad = False

  def unfreeze(self):
    
    """
      A method for recording gradient descent, so that the model weights do not change.
      The goal is for fine tuning or learning with a small learning rate and standard early stopping
    """

    for param in self.resnet50.parameters():
      param.requires_grad = True
