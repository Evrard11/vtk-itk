import itk

def load_image(path: str):
    ImageType = itk.Image[itk.F, 3]
    reader = itk.ImageFileReader[ImageType].New()
    reader.SetFileName(path)
    reader.Update()
    return reader.GetOutput()

def downsample(image):
    ImageType = type(image)
    shrink = itk.ShrinkImageFilter[ImageType, ImageType].New(
        Input=image,
        ShrinkFactors=[3, 3, 3]
    )
    shrink.Update()
    return shrink.GetOutput()

def print_image_info(image, name="Image"):
    region = image.GetLargestPossibleRegion()
    size = region.GetSize()
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    direction = image.GetDirection()
    print(f"--- {name} ---")
    print(f"Size: {size}")
    print(f"Spacing: {spacing}")
    print(f"Origin: {origin}")
    print(f"Direction: {direction}\n")
