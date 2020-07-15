 
"""
Created on Sat Nov 18 23:12:08 2017

@author: Utku Ozbulak - github.com/utkuozbulak

Modified by WooJu Lee
"""
import os
import numpy as np

import torch
from torch.optim import SGD
from torchvision import models

from .misc_functions import preprocess_image, recreate_image, save_image
from PIL import Image
from PIL import ImageFilter

__all__ = [
    "CNNLayerVisualization",
]
class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
        cnn_layer = "layer1"
        block_pos = 0
        sub_layer = "conv1"
        filter_pos = 5
    """
    def __init__(self, model, selected_layer, selected_block, selected_sublayer, selected_filter):

        self.model = model
        self.model.eval()
        self.model_name = model.__class__.__name__

        self.selected_layer = selected_layer
        self.selected_block = selected_block
        self.selected_sublayer = selected_sublayer
        self.selected_filter = selected_filter
        self.conv_output = 0

        self.size = 96
        self.initial_learning_rate = 100
        self.upscaling_factor = 1.2
        self.upscaling_steps = 12
        self.iteration_steps = 20
        self.blur = True

        # Create the folder to export images if not exists
        if not os.path.exists('../generated'):
            os.makedirs('../generated')

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]
        # Hook the selected layer
        for n, m in self.model.named_modules():
            if n == str(self.selected_layer):
                for i, j in m[self.selected_block].named_modules():
                    if i == str(self.selected_sublayer):
                        j.register_forward_hook(hook_function)

        # self.model[self.selected_layer].register_forward_hook(hook_function)

    def visualise_layer_with_hooks(self):

        # Hook the selected layer
        self.hook_layer()

        # Generate a random image
        sz = self.size
        self.created_image = np.uint8(np.random.uniform(150, 180, (sz, sz, 3)))

        for i in range(self.upscaling_steps):
            # Process image and return variable
            self.processed_image = preprocess_image(self.created_image, False)
            # Define optimizer for the image
            optimizer = SGD([self.processed_image], lr=self.initial_learning_rate)

            for j in range(self.iteration_steps):
                optimizer.zero_grad()
                # Assign create image to a variable to move forward in the model
                output = self.model(self.processed_image)

                # Loss function is the mean of the output of the selected layer/filter
                # We try to minimize the mean of the output of that specific filter
                loss = -torch.mean(self.conv_output)
                print('Upscaling:', str(i+1), 'Iteration:', str(j+1), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
                # Backward
                loss.backward()
                # Update image
                optimizer.step()

            # Recreate image
            self.created_image = recreate_image(self.processed_image)
            # Upscale the image
            sz = int(sz * self.upscaling_factor)
            self.created_image = Image.fromarray(self.created_image)
            self.created_image = self.created_image.resize((sz, sz), resample=Image.BICUBIC)
            self.created_image = self.created_image.filter(ImageFilter.BoxBlur(radius=1))

        # Save image
        if (i+1) % 6 == 0:
            # Save image
            im_path = '../generated/layer/' + str(self.model_name) + '_'+ str(self.selected_layer) + '_b' + \
                      str(self.selected_block) + '_' + str(self.selected_sublayer) + '_f' + \
                      str(self.selected_filter) + '_size' + str(self.size) + '_up' + str(i + 1) + '_blur' + '_lr' + \
                      str(self.initial_learning_rate) + '.jpg'
            save_image(self.created_image, im_path)


if __name__ == '__main__':

    # ResNet architecture

    # ResNet(
    #     (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (relu): ReLU(inplace=True)
    # (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    # (layer1): Sequential(
    #     (0): BasicBlock(
    #     (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (relu): ReLU(inplace=True)
    # (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # )
    # (1): BasicBlock(
    #     (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (relu): ReLU(inplace=True)
    # (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # )
    # (2): BasicBlock(
    #     (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (relu): ReLU(inplace=True)
    # (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # )
    # )
    # (layer2): Sequential(
    #     (0): BasicBlock(
    #     (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    # (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (relu): ReLU(inplace=True)
    # (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (downsample): Sequential(
    #     (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
    # (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # )
    # )
    # (1): BasicBlock(
    #     (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (relu): ReLU(inplace=True)
    # (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # )
    # (2): BasicBlock(
    #     (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (relu): ReLU(inplace=True)
    # (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # )
    # (3): BasicBlock(
    #     (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (relu): ReLU(inplace=True)
    # (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # )
    # )
    # (layer3): Sequential(
    #     (0): BasicBlock(
    #     (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    # (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (relu): ReLU(inplace=True)
    # (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (downsample): Sequential(
    #     (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
    # (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # )
    # )
    # (1): BasicBlock(
    #     (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (relu): ReLU(inplace=True)
    # (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # )
    # (2): BasicBlock(
    #     (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (relu): ReLU(inplace=True)
    # (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # )
    # (3): BasicBlock(
    #     (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (relu): ReLU(inplace=True)
    # (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # )
    # (4): BasicBlock(
    #     (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (relu): ReLU(inplace=True)
    # (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # )
    # (5): BasicBlock(
    #     (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (relu): ReLU(inplace=True)
    # (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # )
    # )
    # (layer4): Sequential(
    #     (0): BasicBlock(
    #     (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    # (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (relu): ReLU(inplace=True)
    # (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (downsample): Sequential(
    #     (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
    # (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # )
    # )
    # (1): BasicBlock(
    #     (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (relu): ReLU(inplace=True)
    # (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # )
    # (2): BasicBlock(
    #     (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (relu): ReLU(inplace=True)
    # (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # )
    # )
    # (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
    # (fc): Linear(in_features=512, out_features=1000, bias=True)
    # )

    # ResNet34 architecture
    # (conv1)-(bn1)-(relu)-(maxpool)-(layer1)-(layer2)-(layer3)-(layer4)
    #
    cnn_layer = "layer4"
    block_pos = 2
    sub_layer = "conv2"
    filter_pos = 5
    # Fully connected layer is not needed
    pretrained_model = models.resnet34(pretrained=True).eval()
    # res_layers = list(pretrained_model.children())
    layer_vis = CNNLayerVisualization(pretrained_model, cnn_layer, block_pos, sub_layer, filter_pos)

    # Layer visualization with pytorch hooks
    layer_vis.visualise_layer_with_hooks()

    # Layer visualization without pytorch hooks
    # layer_vis.visualise_layer_without_hooks()