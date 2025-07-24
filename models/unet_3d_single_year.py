'''
Purpose of this python file:
- Define the Unet 3D Architecture
- Include both the original and reduced parameter models
'''

import torch
from torch import nn
import torch.nn.functional as F

# Define a class for a double convolutional block
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, reduce=True, num_groups=32, reduce_kernel_size=3):
        super(DoubleConv, self).__init__()
        # Adjust the number of groups for GroupNorm based on out_ch
        num_groups = min(16, out_ch)
        # Define convolutional layers with group normalization and ReLU activation
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=(3,3,3), padding=1),
            nn.GroupNorm(num_groups, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=(reduce_kernel_size,3,3), padding=(0,1,1)) if reduce else nn.Conv3d(out_ch, out_ch, kernel_size=(reduce_kernel_size,3,3), padding=((reduce_kernel_size-1)//2,1,1)),
            nn.GroupNorm(num_groups, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Forward pass through the convolutional layers
        x = self.conv(x)
        return x

# Define a class for the upsampling operation
class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        # Define the upsampling operation using transposed convolution
        self.up_scale = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=(1,2,2), stride=(1,2,2))

    def forward(self, x1, x2):
        # Upsample the input tensor
        x2 = self.up_scale(x2)

        # Calculate the difference in dimensions between x1 and x2
        diffZ = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        diffX = x1.size()[4] - x2.size()[4]

        # Pad x2 to match the dimensions of x1
        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])

        # Concatenate x1 and x2 along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return x

# Define a class for the downsampling operation
class DownLayer(nn.Module):
    def __init__(self, in_ch, out_ch, reduce, reduce_kernel_size):
        super(DownLayer, self).__init__()
        # Define a max pooling layer and a double convolutional block
        self.pool = nn.MaxPool3d((1,2,2), stride=(1,2,2), padding=0)
        self.conv = DoubleConv(in_ch, out_ch, reduce=reduce, reduce_kernel_size=reduce_kernel_size)

    def forward(self, x):
        # Forward pass through the max pooling and convolutional layers
        x = self.conv(self.pool(x))
        return x


# Define a class for the upsampled layer
class UpLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpLayer, self).__init__()
        # Define the upsampling and double convolutional blocks
        self.up = Up(in_ch, out_ch)
        self.conv = DoubleConv(in_ch, out_ch, reduce=False)

    def forward(self, x1, x2):
        # Perform the upsampling operation
        a = self.up(x1, x2)
        # Forward pass through the convolutional layers
        x = self.conv(a)
        return x
    
    
# Define a class for a double convolutional block
class DoubleConv2(nn.Module):
    def __init__(self, in_ch, out_ch, reduce=True, num_groups=32):
        super(DoubleConv2, self).__init__()
        # Adjust the number of groups for GroupNorm based on out_ch
        num_groups = min(16, out_ch)
        # Define convolutional layers with group normalization and ReLU activation
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=(3,3,3), padding=1),
            nn.GroupNorm(num_groups, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=(3,3,3), padding=(0,1,1)) if reduce else nn.Conv3d(out_ch, out_ch, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.GroupNorm(num_groups, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Forward pass through the convolutional layers
        x = self.conv(x)
        return x

# Define a class for the upsampling operation
class Up2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up2, self).__init__()
        # Define the upsampling operation using transposed convolution
        self.up_scale = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=(1,2,2), stride=(1,2,2))

    def forward(self, x1, x2):
        # Upsample the input tensor
        x2 = self.up_scale(x2)

        # Calculate the difference in dimensions between x1 and x2
        diffZ = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        diffX = x1.size()[4] - x2.size()[4]

        # Pad x2 to match the dimensions of x1
        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])

        # Concatenate x1 and x2 along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return x

# Define a class for the downsampling operation
class DownLayer2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownLayer2, self).__init__()
        # Define a max pooling layer and a double convolutional block
        self.pool = nn.MaxPool3d((1,2,2), stride=(1,2,2), padding=0)
        self.conv = DoubleConv2(in_ch, out_ch, reduce=True)

    def forward(self, x):
        # Forward pass through the max pooling and convolutional layers
        x = self.conv(self.pool(x))
        return x

# Define a class for the upsampled layer
class UpLayer2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpLayer2, self).__init__()
        # Define the upsampling and double convolutional blocks
        self.up = Up2(in_ch, out_ch)
        self.conv = DoubleConv2(in_ch, out_ch, reduce=False)

    def forward(self, x1, x2):
        # Perform the upsampling operation
        a = self.up(x1, x2)
        # Forward pass through the convolutional layers
        x = self.conv(a)
        return x
    
    
# Define the original U-Net architecture
class UNetSixMonth(nn.Module):
    def __init__(self, n_channels: int, dimensions=1):
        super(UNetSixMonth, self).__init__()
        # Define the contracting (downsampling) path of the U-Net
        self.conv1 = DoubleConv(n_channels, 64, reduce=True, reduce_kernel_size=3)  # Starting with 64 channels
        self.down1 = DownLayer(64, 128, reduce=False, reduce_kernel_size=1)
        self.down2 = DownLayer(128, 256, reduce=True, reduce_kernel_size=3)
        self.down3 = DownLayer(256, 512, reduce=False, reduce_kernel_size=1)
        self.down4 = DownLayer(512, 1024, reduce=False, reduce_kernel_size=1)

        # Define the expanding (upsampling) path of the U-Net
        self.up1 = UpLayer(1024, 512)
        self.up2 = UpLayer(512, 256)
        self.up3 = UpLayer(256, 128)
        self.up4 = UpLayer(128, 64)

        # Define the final convolutional layer
        self.last_conv = nn.Conv3d(64, dimensions, 1)
        self.x1_conv = nn.Conv3d(64, 64, kernel_size=(4,1,1), padding=(0,0,0))
        self.x2_conv = nn.Conv3d(128, 128, kernel_size=(4,1,1), padding=(0,0,0))
        self.x3_conv = nn.Conv3d(256, 256, kernel_size=(2,1,1), padding=(0,0,0))
        self.x4_conv = nn.Conv3d(512, 512, kernel_size=(2,1,1), padding=(0,0,0))
        self.x5_conv = nn.Conv3d(1024, 1024, kernel_size=(2,1,1), padding=(0,0,0))

    def forward(self, x):
        # Forward pass through the U-Net architecture
        x1 = self.conv1(x)
        x1_agg = self.x1_conv(x1)

        x2 = self.down1(x1)
        x2_agg = self.x2_conv(x2)

        x3 = self.down2(x2)
        x3_agg = self.x3_conv(x3)

        x4 = self.down3(x3)
        x4_agg = self.x4_conv(x4)

        x5 = self.down4(x4)
        x5_agg = self.x5_conv(x5)

        x1_up = self.up1(x4_agg, x5_agg)
        x2_up = self.up2(x3_agg, x1_up)
        x3_up = self.up3(x2_agg, x2_up)
        x4_up = self.up4(x1_agg, x3_up)

        output = self.last_conv(x4_up)

        # Remove one dimension to match the label data format
        output_tensor = torch.squeeze(output, dim=1)
        return output_tensor
    
    
class UNetTwelveMonth(nn.Module):
    def __init__(self, n_channels: int):
        super(UNetTwelveMonth, self).__init__()
        # Define the contracting (downsampling) path of the U-Net
        self.conv1 = DoubleConv2(n_channels, 64)  # Starting with 64 channels
        self.down1 = DownLayer2(64, 128)
        self.down2 = DownLayer2(128, 256)
        self.down3 = DownLayer2(256, 512)
        self.down4 = DownLayer2(512, 1024)

        # Define the expanding (upsampling) path of the U-Net
        self.up1 = UpLayer2(1024, 512)
        self.up2 = UpLayer2(512, 256)
        self.up3 = UpLayer2(256, 128)
        self.up4 = UpLayer2(128, 64)

        # Define the final convolutional layer
        self.last_conv = nn.Conv3d(64, 1, 1)
        self.x1_conv = nn.Conv3d(64, 64, kernel_size=(10,1,1), padding=(0,0,0))
        self.x2_conv = nn.Conv3d(128, 128, kernel_size=(8,1,1), padding=(0,0,0))
        self.x3_conv = nn.Conv3d(256, 256, kernel_size=(6,1,1), padding=(0,0,0))
        self.x4_conv = nn.Conv3d(512, 512, kernel_size=(4,1,1), padding=(0,0,0))
        self.x5_conv = nn.Conv3d(1024, 1024, kernel_size=(2,1,1), padding=(0,0,0))

    def forward(self, x):
        # Forward pass through the U-Net architecture
        x1 = self.conv1(x)
        x1_agg = self.x1_conv(x1)

        x2 = self.down1(x1)
        x2_agg = self.x2_conv(x2)

        x3 = self.down2(x2)
        x3_agg = self.x3_conv(x3)

        x4 = self.down3(x3)
        x4_agg = self.x4_conv(x4)

        x5 = self.down4(x4)
        x5_agg = self.x5_conv(x5)

        x1_up = self.up1(x4_agg, x5_agg)
        x2_up = self.up2(x3_agg, x1_up)
        x3_up = self.up3(x2_agg, x2_up)
        x4_up = self.up4(x1_agg, x3_up)

        output = self.last_conv(x4_up)

        # Remove one dimension to match the label data format
        output_tensor = torch.squeeze(output, dim=1)
        return output_tensor


# Function to count the number of parameters in a model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Example usage
if __name__ == "__main__":
    # Initialize the models
    model_original = UNetOriginal(dimensions=1)
    model_reduced = UNetReduced(dimensions=1)

    # Print the number of parameters for each model
    print(f"Number of parameters in the original model: {count_parameters(model_original)}")
    print(f"Number of parameters in the reduced model: {count_parameters(model_reduced)}")
    
    # Number of parameters in the original model: 84984257
    # Number of parameters in the reduced model: 5194465