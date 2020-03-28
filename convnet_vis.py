
from PIL import Image
# import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from torchvision import transforms, models
import imageio
import os


class Visualizer:
    
    def __init__(self):
        self.model = models.vgg19(pretrained=True).features
        self.features = None
        self.save_path = 'images'
        self.image_path = 'im_58.bmp'
        for param in self.model.parameters():
            param.requires_grad_(False)
            
            
    def load_image(self, img_path, img_size=400, shape=None):
        image = Image.open(img_path).convert('RGB')
        if max(image.size) > img_size:
            size = img_size
        else:
            size = max(image.size)
        if shape is not None:
            size = shape
        transform = transforms.Compose([
                            transforms.Resize(size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))])
        return transform(image)[:3,:,:].unsqueeze(0)


    def get_conv_features(self, image, model):
        features = list()
        x = image
        for name, layer in model._modules.items():
            x = layer(x)
            if isinstance(layer, torch.nn.Conv2d):
                features.append(x)
        return features

    def normalize(self, tensor):
        norm_img = list()
        for im in tensor:
            im /= np.max(im)
            norm_img.append(im)
        return norm_img
    
    def inference(self):
        image = self.load_image(self.image_path)
        self.features = self.get_conv_features(image, self.model)
        
    def display_layers(self):
        for imgs in self.features:
            print(imgs.shape)
            
    def plot_and_save(self):
        for ii, tensor in enumerate(self.features):
            np_img = tensor.numpy().squeeze()
            norm_img = self.normalize(np_img)
            rgb = (np_img*255).astype(np.uint8)
            for i in range(0, 9):
                image_path  = os.path.join(self.save_path, chr(ii + 97)+'_'+str(i)+'.jpg')
                cv2.imwrite(image_path, rgb[i])

    def make_gif(self):
        image_path = sorted(os.listdir(self.save_path))
        images = list()
        for filename in image_path:
            img = cv2.imread(os.path.join(self.save_path,filename), cv2.IMREAD_GRAYSCALE)
            img = cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)
            images.append(cv2.resize(img, (600, 400)))
        imageio.mimsave('movie.gif', images, duration=0.3)


if __name__ == "__main__":
    vis = Visualizer()
    vis.inference()
    vis.plot_and_save()
    vis.make_gif()
    