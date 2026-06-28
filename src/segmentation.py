import itk
import numpy as np
from scipy import ndimage as ndi


def _array_to_itk(arr, reference):
    img = itk.image_from_array(arr)
    img.SetSpacing(reference.GetSpacing())
    img.SetOrigin(reference.GetOrigin())
    img.SetDirection(reference.GetDirection())
    return img


def _border_mask(shape):
    b = np.zeros(shape, dtype=bool)
    b[0, :, :] = b[-1, :, :] = True
    b[:, 0, :] = b[:, -1, :] = True
    b[:, :, 0] = b[:, :, -1] = True
    return b


def segment_tumor(arr, bg=50, dark_upper=200):
    arr_f = arr.astype(float)
    smooth = ndi.gaussian_filter(arr_f, sigma=2)

    # La tumeur est hypointense sur T1 : on cherche les zones sombres dans le corps (bg < intensite < dark_upper)
    dark = (smooth > bg) & (smooth < dark_upper)

    # Erosion pour eliminer les sillons cerebraux fins, la tumeur (large masse compacte) survit
    eroded = ndi.binary_erosion(dark, iterations=5)

    labeled, n = ndi.label(eroded)
    if n == 0:
        return np.zeros_like(arr, dtype=np.uint8)

    border = _border_mask(arr.shape)

    # On exclut les composantes connexes qui touchent le bord (fond, artefacts de recalage)
    candidates = []
    for i in range(1, n + 1):
        comp = labeled == i
        if not np.any(comp & border):
            candidates.append((comp.sum(), i))

    if not candidates:
        return np.zeros_like(arr, dtype=np.uint8)

    best = max(candidates, key=lambda x: x[0])[1]
    core = labeled == best

    # Dilatation pour recuperer les bords erodes, puis fermeture morphologique
    tumor = ndi.binary_dilation(core, iterations=5)
    tumor = ndi.binary_closing(tumor, iterations=3)

    return tumor.astype(np.uint8) * 255


def run_segmentation(fixed, moved):
    arr1 = itk.array_from_image(fixed).astype(float)
    arr2 = itk.array_from_image(moved).astype(float)

    m1 = segment_tumor(arr1)
    m2 = segment_tumor(arr2)

    vol1 = int((m1 > 0).sum())
    vol2 = int((m2 > 0).sum())
    print(f"Volume GRE1 : {vol1} mm3")
    print(f"Volume GRE2 : {vol2} mm3")
    print(f"Delta volume : {vol2 - vol1:+d} mm3")

    itk.imwrite(_array_to_itk(m1, fixed), "data/mask_gre1.nrrd")
    itk.imwrite(_array_to_itk(m2, moved), "data/mask_gre2.nrrd")
    return _array_to_itk(m1, fixed), _array_to_itk(m2, moved)
