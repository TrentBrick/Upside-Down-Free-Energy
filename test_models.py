from models import UpsdModel, UpsdBehavior
import torch 
model = UpsdModel(8,2,4,[128,128,128, 24])
print(model)

states, desires = torch.ones((50,8)), torch.zeros((50,2))

print( model(states,desires) )
print(model.output_fc.bias)