import torchvision
from torchvision import transforms
def transforms_compose(steps, include_random=True): # Compose the transform steps
    if steps[0][0] == 'BGR2RGB':
        BGR2RGB = True
        steps = steps[1:]
    else:
        BGR2RGB = False       
    
    transform_list = []
    if BGR2RGB == True:
        transform_list.append(transforms.Lambda(lambda img: img[:,:,::-1]))
    transform_list.append(transforms.ToPILImage())
    for transform, inp in steps:
        if transform.startswith('Random') and include_random == False:
            continue
        if transform == 'RandomHorizontalFlip':
            transform_list.append(transforms.RandomHorizontalFlip())
        elif transform == 'RandomResizedCrop':
            transform_list.append(transforms.RandomResizedCrop(size=inp['size'], scale=inp['scale'], ratio=inp['ratio']))
        elif transform == 'Resize':
            transform_list.append(transforms.Resize((inp, inp)))
        else:
            raise ValueError('Unrecognized transformation.')
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)