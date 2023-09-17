from src.model_arch import resnet18
from torchsummary import summary

# Untrained model
# my_model = vgg11_bn()
# my_model = restnet18()

# Pretrained model
my_model = resnet18(pretrained=True)
my_model.eval() # for evaluation
# print(my_model)
summary(model=my_model, input_size=(3, 224, 224), batch_size=1, device='cpu')