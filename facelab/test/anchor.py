from facelab.util.anchor import prior_box

priors = prior_box(image_sizes = (224, 224),
                   min_sizes = [[16, 32], [64, 128], [256, 512]],
                   steps = [8, 16, 32])
print(priors)
print(priors.shape)
