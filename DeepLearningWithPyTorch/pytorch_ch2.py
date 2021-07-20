
import torch

# create a random "image" 5x5 with 3 color channels 
img_t = torch.randn(3,5,5)

# create an "image" batch of 5x5 images with 3 color channels 
# i.e 2 5x5 images
batch_t = torch.randn(2,3,5,5)

# some weights
weights = torch.tensor([0.2126, 0.7152, 0.0722])

# notice that in the image and batch example, 
# the channels dimension is either 0 or 1, 
# but is always third from the end
# Then to get the mean over the channels we can do this

print(f"Taking the mean over channel, we have a resulting shape of {img_t.mean(-3).shape}\n")

# now lets take the weights and broadcast them over the images to get
# gray-scale images

# unsqueeze at the end twice to broadcast one weight per channel.
unsqueezed_weights = weights.unsqueeze(-1).unsqueeze(-1)
print("Unsqueezed Weights")
print(unsqueezed_weights, "")

# Then we can multiply the weights by the image slices to convert to gray-scale
img_weighted = img_t * unsqueezed_weights
print(f"\nThe resulting tensor has shape {img_weighted.shape}")
# Then sum across the channel dimension to get gray-scale
img_scaled = img_weighted.sum(-3)
print(f"After summing over the channels, the image has shape {img_scaled.shape}\n")

# Naming Channels

# construct a tensor with weights
weights_named = torch.tensor([0.2126, 0.7152, 0.0722], names=['channels'])

# add names to a tensor
img_named = img_t.refine_names(..., "channels", "rows", "columns")

# align weights to an image tensor
weights_aligned = weights_named.align_as(img_named)
# [3, 1, 1]

# recreate grey-scaled image with named tensors
grey_named = (img_named * weights_aligned).sum('channels')
print()
print(grey_named)

# to get back to unnamed
grey_unnamed = grey_named.rename(None)
print()
print(grey_unnamed)

