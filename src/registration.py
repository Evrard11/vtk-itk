import itk

def translation_registration(fixed, moving):
    ImageType = itk.Image[itk.F, 3]
    RegistrationType = itk.ImageRegistrationMethodv4[ImageType, ImageType]
    registration = RegistrationType.New()
    # transformation
    TransformType = itk.TranslationTransform[itk.D, 3]
    transform = TransformType.New()
    transform.SetIdentity()
    registration.SetInitialTransform(transform)
    # metric
    MetricType = itk.MeanSquaresImageToImageMetricv4[ImageType, ImageType]
    metric = MetricType.New()
    registration.SetMetric(metric)
    # optimizer
    OptimizerType = itk.RegularStepGradientDescentOptimizerv4[itk.D]
    optimizer = OptimizerType.New()
    optimizer.SetLearningRate(4)
    optimizer.SetMinimumStepLength(0.01)
    optimizer.SetNumberOfIterations(100)
    # registration parameters
    registration.SetOptimizer(optimizer)
    registration.SetFixedImage(fixed)
    registration.SetMovingImage(moving)
    registration.Update()
    print("Translation registration done")
    #print(registration.GetMetric())
    return registration.GetTransform()

def rigid_registration(fixed, moving):
    ImageType = itk.Image[itk.F, 3]
    registration = itk.ImageRegistrationMethodv4[ImageType, ImageType].New()
    # transform
    transform = itk.VersorRigid3DTransform[itk.D].New()
    # init transform at center for rotation
    initializer = itk.CenteredTransformInitializer[
        itk.VersorRigid3DTransform[itk.D], ImageType, ImageType
    ].New()
    initializer.SetTransform(transform)
    initializer.SetFixedImage(fixed)
    initializer.SetMovingImage(moving)
    initializer.GeometryOn()
    initializer.InitializeTransform()
    registration.SetInitialTransform(transform)
    registration.InPlaceOn()
    # metric
    metric = itk.MattesMutualInformationImageToImageMetricv4[ImageType, ImageType].New()
    metric.SetNumberOfHistogramBins(50)
    registration.SetMetric(metric)
    # optimizer
    optimizer = itk.RegularStepGradientDescentOptimizerv4[itk.D].New()
    optimizer.SetLearningRate(2.0)
    optimizer.SetMinimumStepLength(1e-4)
    optimizer.SetNumberOfIterations(200)
    optimizer.SetRelaxationFactor(0.5)
    # set different scales learning rates according to rotation (rad) or translation (mm)
    scales = itk.OptimizerParameters[itk.D](6)
    scales[0] = 1.0
    scales[1] = 1.0
    scales[2] = 1.0

    scales[3] = 0.01
    scales[4] = 0.01
    scales[5] = 0.01
    optimizer.SetScales(scales)
    # registration parameters
    registration.SetOptimizer(optimizer)
    registration.SetFixedImage(fixed)
    registration.SetMovingImage(moving)
    registration.SetNumberOfLevels(1)
    registration.Update()
    print("Rigid registration done")
    return registration.GetTransform()

def resample_image(fixed, moving, transform):
    ResampleFilter = itk.ResampleImageFilter[itk.Image[itk.F, 3], itk.Image[itk.F, 3]].New()
    ResampleFilter.SetInput(moving)
    ResampleFilter.SetTransform(transform)
    ResampleFilter.SetReferenceImage(fixed)
    ResampleFilter.UseReferenceImageOn()
    ResampleFilter.Update()
    return ResampleFilter.GetOutput()
