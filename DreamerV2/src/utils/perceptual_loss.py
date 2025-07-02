
import torch
import torch.nn as nn
import torchvision.models as models

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        vgg.eval() 

        for param in vgg.parameters():
            param.requires_grad = False

        self.feature_layers = nn.Sequential(
            vgg[0],  # Conv1_1
            vgg[1],  # ReLU1_1
            vgg[2],  # Conv1_2
            vgg[3],  # ReLU1_2
            vgg[4],  # MaxPool1
            vgg[5],  # Conv2_1
            vgg[6],  # ReLU2_1
            vgg[7],  # Conv2_2
            vgg[8],  # ReLU2_2
            vgg[9],  # MaxPool2
            vgg[10], # Conv3_1
            vgg[11], # ReLU3_1
            vgg[12], # Conv3_2
            vgg[13], # ReLU3_2
            vgg[14], # Conv3_3
            vgg[15], # ReLU3_3
            vgg[16]  # MaxPool3
        ).to(next(vgg.parameters()).device) # Move to the same device as VGG

        self.loss = nn.MSELoss()
        self.resize = resize

    def forward(self, x, y):
        """
        Computes the perceptual loss between two batches of images.
        Args:
            x (torch.Tensor): The reconstructed images, shape (N, C, H, W).
                              Expected pixel values in [-0.5, 0.5].
            y (torch.Tensor): The ground truth images, shape (N, C, H, W).
                              Expected pixel values in [-0.5, 0.5].
        """
        # VGG expects 3-channel images normalized according to ImageNet standards.
        # Our images are grayscale (stacked as channels) and in [-0.5, 0.5].
        # We'll convert them to 3-channel and apply the normalization.

        # 1. Convert from [-0.5, 0.5] to [0, 1]
        x = x + 0.5
        y = y + 0.5

        # 2. If the input is single-channel, repeat it to make it 3-channel
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        # If we have 4 stacked frames, we can take the last one and make it 3-channel
        elif x.shape[1] == 4:
            x = x[:, -1, :, :].unsqueeze(1).repeat(1, 3, 1, 1)
            y = y[:, -1, :, :].unsqueeze(1).repeat(1, 3, 1, 1)


        # 3. (Optional but recommended) Resize images to what VGG expects (e.g., 224x224)
        if self.resize:
            x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            y = nn.functional.interpolate(y, size=(224, 224), mode='bilinear', align_corners=False)

        # 4. Normalize using ImageNet mean and std
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        y = (y - mean) / std

        # 5. Extract features and compute loss
        x_features = self.feature_layers(x)
        y_features = self.feature_layers(y)

        return self.loss(x_features, y_features)
