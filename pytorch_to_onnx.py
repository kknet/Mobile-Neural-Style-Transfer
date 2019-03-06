# coding=UTF-8
from transformer_net import TransformerNet
from torchvision import transforms
import torch
import sys
import re
from PIL import Image

def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img

style_model = TransformerNet()
state_dict = torch.load(sys.argv[1])

# remove saved deprecated running_* keys in InstanceNorm from the checkpoint
for k in list(state_dict.keys()):
    if re.search(r'in\d+\.running_(mean|var)$', k):
        del state_dict[k]
style_model.load_state_dict(state_dict)


content_file = "images/content-images/amber.jpg"
content_image = load_image(content_file)
content_transform = transforms.Compose([
    transforms.ToTensor(),
])
content_image = content_transform(content_image)
content_image = content_image.unsqueeze(0)

assert sys.argv[2].endswith(".onnx"), "Export model file should end with .onnx"
torch.onnx.export(style_model, 
                  content_image, 
                  sys.argv[2],
                  verbose=True,
                  export_params=True,
                  input_names=['inputImage'], 
                  output_names=['outputImage'])
