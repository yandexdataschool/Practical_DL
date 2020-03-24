[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yandexdataschool/Practical_DL/blob/spring20/seminar06-style-transfer/style_transfer_pytorch.ipynb)



`to be used in seminar`

```(python)
import multiprocessing as mp
import multiprocessing.shared_memory
import numpy as np
import torch
from pytorch_pretrained_biggan import BigGAN
import matplotlib.pyplot as plt
%matplotlib inline
model = BigGAN.from_pretrained('biggan-deep-128').cuda().eval()
model.cuda();
model();
with torch.no_grad():

    z_torch = torch.randn(1, 128)
    c_torch = torch.randint(0, 1000, size=[1])

    gen_images = model.forward(
        z_torch.clamp_(-5, 5).cuda(), 
        torch.nn.functional.one_hot(c_torch.clamp_(0, 999).cuda(), num_classes=1000).to(torch.float32),
        truncation=0.4)

    gen_images = torch.clamp((gen_images + 1) / 2, 0, 1)
    image = gen_images.data.cpu().numpy()[0].transpose(1, 2, 0)

    plt.imshow(image)
```
