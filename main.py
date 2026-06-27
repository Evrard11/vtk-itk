import itk

from src.load_data import load_image, downsample, print_image_info
from src.registration import rigid_registration, rigid_registration, resample_image
from src.registration_utils import get_rmse, show_slices, show_overlap_slices

def main_registration():
    fixed = load_image("data/case6_gre1.nrrd")
    moving = load_image("data/case6_gre2.nrrd")
    print_image_info(fixed, "Fixed")
    print_image_info(moving, "Moving")

    # downspample images to go faster
    rigid_tx = rigid_registration(downsample(fixed), downsample(moving))
    moved_rigid = resample_image(fixed, moving, rigid_tx)

    # cast to get lighter file (original files already using short type)
    caster = itk.CastImageFilter[itk.Image[itk.F,3], itk.Image[itk.SS, 3]].New(Input=moved_rigid)
    itk.imwrite(caster.GetOutput(), "data/rigid_registered.nrrd", compression=True)

    print(f"RMSE Before: {get_rmse(fixed, moving)}, RMSE After: {get_rmse(fixed, moved_rigid)}")
    # show registration
    show_slices(fixed, moved_rigid)
    show_overlap_slices(fixed, moved_rigid)
    
if __name__ == "__main__":
    main_registration()
