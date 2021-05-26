import torch

import fgbg


model = fgbg.AutoEncoder(
    feature_size=512,
    projected_size=512,
    input_channels=3,
    decode_from_projection=True,
)

feature = model.encoder(torch.rand(1, 3, 200, 200))
print(feature.shape)
print(model.decoder(feature).shape)
