import itk

from src.load_data import load_image, downsample, print_image_info
from src.registration import translation_registration, rigid_registration, affine_registration, b_spline_registration, resample_image
from src.registration_utils import get_rmse, show_slices, show_overlap_slices
from src.segmentation import run_segmentation
from src.visualisation import (
    compute_changes,
    compute_metrics,
    render_change_overlay
)

def main_registration():
    fixed = load_image("data/case6_gre1.nrrd")
    moving = load_image("data/case6_gre2.nrrd")
    print_image_info(fixed, "Fixed")
    print_image_info(moving, "Moving")

    import os
    registered_path = "data/case6_gre2_registered.nrrd"
    if os.path.exists(registered_path):
        print("Recalage déjà effectué, chargement du fichier existant.")
        moved = load_image(registered_path)
    else:
        d_fixed, d_moving = downsample(fixed), downsample(moving)
        rigid_tx = rigid_registration(d_fixed, d_moving)
        affine_tx = affine_registration(d_fixed, d_moving, initial_transform=rigid_tx)
        bspline_tx = b_spline_registration(d_fixed, d_moving, initial_transform=affine_tx)
        moved = resample_image(fixed, moving, bspline_tx)
        caster = itk.CastImageFilter[itk.Image[itk.F,3], itk.Image[itk.SS, 3]].New(Input=moved)
        itk.imwrite(caster.GetOutput(), registered_path, compression=True)
        print(f"RMSE Before: {get_rmse(fixed, moving)}, RMSE After: {get_rmse(fixed, moved)}")
        # show_slices(fixed, moved)
        # show_overlap_slices(fixed, moved)

    # Segmentation
    mask_1, mask_2 = run_segmentation(fixed, moved)

    # Visualisation
    added, removed, stable = compute_changes(mask_1, mask_2)
    metrics = compute_metrics(mask_1, mask_2, fixed, moved)
    render_change_overlay(fixed,added,removed,stable,metrics)

if __name__ == "__main__":
    main_registration()
    
