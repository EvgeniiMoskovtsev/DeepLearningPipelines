from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as tr

from modeling.deeplab import DeepLab
from dataloaders.utils import decode_segmap

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load("run/pascal/deeplab-resnet/experiment_12/checkpoint.pth.tar")

model = DeepLab(num_classes=2,
                backbone='resnet',
                output_stride=16,
                sync_bn=True,
                freeze_bn=False)

model.load_state_dict(checkpoint['state_dict'])
model.eval()
model.to(device)

def transform(image):
    return tr.Compose([
        tr.Resize(513),
        tr.CenterCrop(513),
        tr.ToTensor(),
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])(image)

torch.set_grad_enabled(False)

image = cv2.imread('/home/evg/Documents/All_segmentations/Qubvel/voc_1/train/JPEGImages/1371_102.png')
image = Image.fromarray(image)
inputs = transform(image).to(device)
output = model(inputs.unsqueeze(0)).squeeze().cpu().numpy()
# plt.imshow(output[0])
# plt.show()
# plt.imshow(output[1])
# plt.show()
pred = np.argmax(output, axis=0)
plt.imshow(pred)
plt.show()
# Then visualize it:
# decode_segmap(pred, dataset="pascal", plot=True)


