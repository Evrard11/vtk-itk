import itk

def translation_registration(fixed, moving):
    ImageType = itk.Image[itk.F, 3]
    RegistrationType = itk.ImageRegistrationMethodv4[ImageType, ImageType]
    registration = RegistrationType.New()

    TransformType = itk.TranslationTransform[itk.D, 3]
    transform = TransformType.New()
    transform.SetIdentity()
    registration.SetInitialTransform(transform)

    MetricType = itk.MeanSquaresImageToImageMetricv4[ImageType, ImageType]
    metric = MetricType.New()
    registration.SetMetric(metric)

    OptimizerType = itk.RegularStepGradientDescentOptimizerv4[itk.D]
    optimizer = OptimizerType.New()
    optimizer.SetLearningRate(4)
    optimizer.SetMinimumStepLength(0.01)
    optimizer.SetNumberOfIterations(100)
    registration.SetOptimizer(optimizer)
    registration.SetFixedImage(fixed)
    registration.SetMovingImage(moving)
    registration.Update()
    print("Translation done")
    #print(registration.GetMetric())
    return registration.GetTransform()

def resample_image(fixed, moving, transform):
    ResampleFilter = itk.ResampleImageFilter[itk.Image[itk.F, 3], itk.Image[itk.F, 3]].New()
    ResampleFilter.SetInput(moving)
    ResampleFilter.SetTransform(transform)
    ResampleFilter.SetReferenceImage(fixed)
    ResampleFilter.UseReferenceImageOn()
    ResampleFilter.Update()
    return ResampleFilter.GetOutput()
