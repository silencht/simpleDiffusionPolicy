# for standard resnet
from typing import Callable
import torch
import torch.nn as nn
import torchvision

# for robomimic observation encoder, refer https://github.com/ARISE-Initiative/robomimic/blob/master/examples/simple_obs_nets.py
from collections import OrderedDict
from robomimic.models.base_nets import MLP
from robomimic.models.obs_nets import ObservationEncoder, ObservationDecoder
from robomimic.models.obs_core import CropRandomizer
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils


#  1. Standard Resnet
#  Defines helper functions:
#  - `get_resnet` to initialize standard ResNet vision encoder
#  - `replace_bn_with_gn` to replace all BatchNorm layers with GroupNorm

def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module

def get_resnet(name:str, weights=None, **kwargs) -> nn.Module:
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", None
    """
    # Use standard ResNet implementation from torchvision
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)

    # remove the final fully connected layer
    # for resnet18, the output dim should be 512
    resnet.fc = torch.nn.Identity()

    return resnet

#  2. Robomimic: Resnet18 + spacial softmax
def get_robomimic_resnet(camera_shape=[3, 96, 96], 
                         backbone_class="ResNet18Conv", 
                         pretrained_backbone=False, 
                         input_coord_conv=False, 
                         pool_class="SpatialSoftmax", 
                         num_kp=32, 
                         feature_dimension=512):
    """
    Constructs an observation encoder using a ResNet18Conv backbone and SpatialSoftmax pooling.

    Args:
        camera_shape (list): The shape of the input image, e.g., [3, 224, 224]. Default is [3, 224, 224].
        backbone_class (str): The backbone class for visual processing. Default is "ResNet18Conv".
        pretrained_backbone (bool): Whether to use a pretrained backbone. Default is False.
        input_coord_conv (bool): Whether to use input coordinate convolution. Default is False.
        pool_class (str): The pooling class. Default is "SpatialSoftmax".
        num_kp (int): The number of keypoints for pooling. Default is 32.
        feature_dimension (int): The output feature dimension. Default is 64.

    Returns:
        ObservationEncoder: The constructed observation encoder.
    """
    obs_encoder = ObservationEncoder(feature_activation=torch.nn.ReLU)

    net_kwargs = {
        "input_shape": camera_shape,
        "backbone_class": backbone_class,
        "backbone_kwargs": {"pretrained": pretrained_backbone, "input_coord_conv": input_coord_conv},
        "pool_class": pool_class,
        "pool_kwargs": {"num_kp": num_kp},
        "feature_dimension": feature_dimension,
    }

    # We will use a reconfigurable image processing backbone VisualCore to process the input image observation key
    net_class = "VisualCore"  # this is defined in models/base_nets.py

    # register the network for processing the observation key
    obs_encoder.register_obs_key(
        name="camera",
        shape=camera_shape,
        net_class=net_class,
        net_kwargs=net_kwargs,
    )

    # Before constructing the encoder, make sure we register all of our observation keys with corresponding modalities
    # (this will determine how they are processed during training)
    obs_modality_mapping = {
        "rgb": ["camera"],
    }
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping=obs_modality_mapping)

    # Finally, construct the observation encoder
    obs_encoder.make()

    return obs_encoder


## Demo
if __name__ == "__main__":
        # 1. Standard Resnet
        batch = 64
        obs_horizon = 2
        # construc ResNet18 encoder
        # IMPORTANT! replace all BatchNorm with GroupNorm to work with EMA, performance will tank if you forget to do this!
        vision_encoder = get_resnet('resnet18')
        vision_encoder = replace_bn_with_gn(vision_encoder)
        image = torch.zeros((batch,obs_horizon,3,96,96))
        # # vision encoder
        print(image.flatten(end_dim=1).shape) # (128, 3, 96, 96)
        image_features = vision_encoder(image.flatten(end_dim=1))
        print(image_features.shape)           # (2,512)
        image_features = image_features.reshape(*image.shape[:2],-1)
        print(image_features.shape)           # (64,2,512)

        # 2. Robomimic: Resnet18 + spacial softmax
        robomimic_encoder = get_robomimic_resnet()
        robomimic_encoder = replace_bn_with_gn(robomimic_encoder)
        # Construct fake inputs
        image = torch.zeros((batch,3,96,96))
        inputs = {"camera": image,}
        # # Send to GPU if applicable
        if torch.cuda.is_available():
            inputs = TensorUtils.to_device(inputs, torch.device("cuda:0"))
            robomimic_encoder.cuda()
        print("Construct fake inputs, shape is ", inputs["camera"].shape)
        # output from each obs key network is concatenated as a flat vector.
        # The concatenation order is the same as the keys are registered
        obs_feature = robomimic_encoder(inputs)
        print("obs_feature shape is ",obs_feature.shape)