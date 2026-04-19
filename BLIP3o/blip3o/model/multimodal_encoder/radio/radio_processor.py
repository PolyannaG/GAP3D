from timm.data.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers.image_processing_utils import BatchFeature
from PIL import Image
from transformers.image_transforms import convert_to_rgb
import numpy as np

class BaseProcessor:
    def __init__(self):
        self.transform = lambda x: x
        return

    def __call__(self, item):
        return self.transform(item)

class RadioImageBaseProcessor(BaseProcessor):
    def __init__(self,  mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD):
        self.mean = mean if mean is not None else OPENAI_CLIP_MEAN
        self.std = std if std is not None else OPENAI_CLIP_STD
        self.normalize = transforms.Normalize(self.mean, self.std)

    @property
    def image_mean(self):
        return self.mean

class RadioImageTrainProcessor(RadioImageBaseProcessor):
    def __init__(self, image_size, mean=None, std=None):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                convert_to_rgb,
                transforms.Resize(
                    image_size,
                    interpolation=InterpolationMode.BICUBIC, # should we do BILINEAR as in radio example?
                ),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

        self.image_size = image_size
        print(f"RadioImageTrainProcessor initialized with image size: {image_size}")

    def preprocess(self, images, return_tensors):
        if isinstance(images, Image.Image):
            images = [images]
        else:
            assert isinstance(images, list)
        
        
        transformed_images = [self.transform(image).numpy() for image in images]
    
        data = {"pixel_values": transformed_images}

        return BatchFeature(data=data, tensor_type=return_tensors)

    def __call__(self, item):
        return self.transform(item)

    @property
    def crop_size(self):
        return {"height": self.image_size, "width": self.image_size}

    @property
    def size(self):
        return {"shortest_edge": self.image_size}