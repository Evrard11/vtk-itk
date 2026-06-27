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
    # (to delete) level 1 for debug to go faster (but bad results)
    registration.SetNumberOfLevels(1)
    print("\rRigid registration ...", end="")
    registration.Update()
    print("\rRigid registration done")
    return transform

def affine_registration(fixed, moving, initial_transform=None):
    ImageType = itk.Image[itk.F, 3]
    AffineTransformType = itk.AffineTransform[itk.D, 3]
    registration = itk.ImageRegistrationMethodv4[ImageType, ImageType].New()
    # transform
    transform = AffineTransformType.New()
    if initial_transform is not None:
        # warm start from precalculated (rigid) transformation
        transform.SetCenter(initial_transform.GetCenter())
        transform.SetMatrix(initial_transform.GetMatrix())
        transform.SetTranslation(initial_transform.GetTranslation())
    else:
        # otherwise classic centered init
        initializer = itk.CenteredTransformInitializer[AffineTransformType, ImageType, ImageType].New()
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
    optimizer.SetLearningRate(1.0)
    optimizer.SetMinimumStepLength(1e-4)
    optimizer.SetNumberOfIterations(300)
    optimizer.SetRelaxationFactor(0.5)
    # set different scales learning rates according to rotation (rad) or translation (mm)
    scales = itk.OptimizerParameters[itk.D](12)
    for i in range(9):
        scales[i] = 1.0
    for i in range(9, 12):
        scales[i] = 0.001
    optimizer.SetScales(scales)
    # registration parameters
    registration.SetOptimizer(optimizer)
    registration.SetFixedImage(fixed)
    registration.SetMovingImage(moving)
    print("\rAffine registration ...", end="")
    registration.Update()
    print("\rAffine registration done")
    return transform

# initial transform not reoptimized (only b-spline grid)
def b_spline_registration(fixed, moving, initial_transform=None, grid_nodes=8):
    SplineOrder = 3
    ImageType = itk.Image[itk.F, 3]
    BSplineTransformType = itk.BSplineTransform[itk.D, 3, SplineOrder]
    registration = itk.ImageRegistrationMethodv4[ImageType, ImageType].New()
    # transform is the b-spline control grid points
    transform = BSplineTransformType.New()
    MeshSizeType = itk.Size[3]
    mesh_size = MeshSizeType()
    for i in range(3):
        mesh_size[i] = grid_nodes - SplineOrder
    initializer = itk.BSplineTransformInitializer[BSplineTransformType, ImageType].New()
    initializer.SetTransform(transform)
    initializer.SetImage(fixed)
    initializer.SetTransformDomainMeshSize(mesh_size)
    initializer.InitializeTransform()
    # initial transform fixed (not optimized) only b-spline object is optimized
    if initial_transform is not None:
        registration.SetMovingInitialTransform(initial_transform)
    registration.SetInitialTransform(transform)
    registration.InPlaceOn()
    # metric
    metric = itk.MattesMutualInformationImageToImageMetricv4[ImageType, ImageType].New()
    metric.SetNumberOfHistogramBins(50)
    registration.SetMetric(metric)
    # optimizer
    optimizer = itk.LBFGSBOptimizerv4.New()
    num_parameters = transform.GetNumberOfParameters()
    # bound_select parameter = 0 to be unbounded
    bound_select = itk.Array[itk.SL](num_parameters)
    bound_select.Fill(0)
    lower_bound = itk.Array[itk.D](num_parameters)
    lower_bound.Fill(0.0)
    upper_bound = itk.Array[itk.D](num_parameters)
    upper_bound.Fill(0.0)
    optimizer.SetBoundSelection(bound_select)
    optimizer.SetLowerBound(lower_bound)
    optimizer.SetUpperBound(upper_bound)
    optimizer.SetCostFunctionConvergenceFactor(1e+10)  # default: 1e+7 (bigger = faster/less precise)
    optimizer.SetGradientConvergenceTolerance(1e-4)  # default: 1e-5
    optimizer.SetNumberOfIterations(300)
    optimizer.SetMaximumNumberOfFunctionEvaluations(300)
    optimizer.SetMaximumNumberOfCorrections(5)
    # registration parameters
    registration.SetOptimizer(optimizer)
    registration.SetFixedImage(fixed)
    registration.SetMovingImage(moving)
    registration.SetNumberOfLevels(1)
    print(f"\rB-spline registration ({transform.GetNumberOfParameters()} parameters) ...", end="")
    registration.Update()
    print("\rB_spline registration done")
    # final transform affine (fixed) + b-spline
    output_transform = itk.CompositeTransform[itk.D, 3].New()
    if initial_transform is not None:
        output_transform.AddTransform(initial_transform)
    output_transform.AddTransform(transform)
    return output_transform

def resample_image(fixed, moving, transform):
    ResampleFilter = itk.ResampleImageFilter[itk.Image[itk.F, 3], itk.Image[itk.F, 3]].New()
    ResampleFilter.SetInput(moving)
    ResampleFilter.SetTransform(transform)
    ResampleFilter.SetReferenceImage(fixed)
    ResampleFilter.UseReferenceImageOn()
    ResampleFilter.Update()
    return ResampleFilter.GetOutput()
