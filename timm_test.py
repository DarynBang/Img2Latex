import timm
import torch
import torch.nn as nn
from CNN_Architectures.util import CBAM


class EncoderCNN(nn.Module):
    def __init__(self, train_CNN=False):
        super().__init__()
        resnet = timm.create_model(
            'resnetv2_34d.ra4_e3600_r224_in1k',
            pretrained=False,
            features_only=True,
        )
        self.train_CNN = train_CNN

        # Freeze layers
        for param in resnet.parameters():
            param.requires_grad = train_CNN

        self.layers = nn.Sequential(
            resnet.stem_conv1,
            resnet.stem_norm1,

            resnet.stem_conv2,
            resnet.stem_norm2,
            CBAM(32, 4),

            resnet.stem_conv3,
            resnet.stem_pool,
            resnet.stages_0,
            resnet.stages_1,
            resnet.stages_2,
            resnet.stages_3,
        )

    def forward(self, x):
        f = self.layers(x)
        return f

def test_model():
    # Instantiate the model
    encoder = EncoderCNN(train_CNN=False)


    dummy_input = torch.randn(1, 3, 224, 224)  # [batch_size, channels, height, width]

    # Forward pass through the backbone
    output = encoder(dummy_input)

    # Check the type and shape of the output
    print(f"Output type: {type(output)}")
    print(f"Output shape: {output.shape if isinstance(output, torch.Tensor) else [o.shape for o in output]}")


# Check model
def check_model():
    model = timm.create_model('resnetv2_34d.ra4_e3600_r224_in1k',
            pretrained=False,
            features_only=True,)

    print(model)
    print('-----')
    print(model.stem_conv2)
    print(model.stem_norm2)


check_model()



