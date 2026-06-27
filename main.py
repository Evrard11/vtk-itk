import itk

from src.load_data import load_image, downsample, print_image_info
from src.registration import translation_registration, resample_image
from src.registration_utils import get_rmse, show_slices, show_overlap_slices

def main_registration():
    fixed = load_image("data/case6_gre1.nrrd")
    moving = load_image("data/case6_gre2.nrrd")
    print_image_info(fixed, "Fixed")
    print_image_info(moving, "Moving")

    translation_tx = translation_registration(downsample(fixed), downsample(moving))
    moved_translation = resample_image(fixed, moving, translation_tx)

    itk.imwrite(moved_translation, "data/translation_registered.nrrd")
    print(f"RMSE Before: {get_rmse(fixed, moving)}, RMSE After: {get_rmse(fixed, moved_translation)}")
    # show registration
    # show_slices(fixed, moved_translation)
    # show_overlap_slices(fixed, moved_translation)
    
if __name__ == "__main__":
    main_registration()
