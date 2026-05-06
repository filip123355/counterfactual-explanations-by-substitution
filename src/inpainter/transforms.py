from torchvision import transforms

from src.constants import I2SB_IMAGE_SIZE

I2SB_MEAN = (0.5, 0.5, 0.5)
I2SB_STD = (0.5, 0.5, 0.5)

PIL_TO_I2SB = transforms.Compose(
    [
        transforms.Resize((I2SB_IMAGE_SIZE, I2SB_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=I2SB_MEAN, std=I2SB_STD),
    ]
)

inverse_mean = [-m / s for m, s in zip(I2SB_MEAN, I2SB_STD)]
inverse_std = [1 / s for s in I2SB_STD]

I2SB_TO_NORMAL = transforms.Compose(
    [
        transforms.Normalize(mean=inverse_mean, std=inverse_std),
    ]
)

I2SB_TO_PIL = transforms.Compose(
    [
        I2SB_TO_NORMAL,
        transforms.ToPILImage(),
    ]
)
