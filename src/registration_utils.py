import itk
import matplotlib.pyplot as plt
import numpy as np

def itk_to_numpy(image):
    return itk.GetArrayFromImage(image)

def get_rmse(fixed, moving):
    f = itk_to_numpy(fixed).astype(np.float32)
    m = itk_to_numpy(moving).astype(np.float32)
    f = (f-f.min())/(f.max()-f.min())
    m = (m-m.min())/(m.max()-m.min())
    return np.sqrt(np.mean((f - m)**2))

def show_slices(fixed, moving, slice_idx=None):
    f = itk_to_numpy(fixed)
    m = itk_to_numpy(moving)
    if slice_idx is None:
        slice_idx = f.shape[0] // 2
    fixed_slice = f[slice_idx, :, :]
    moving_slice = m[slice_idx, :, :]
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Fixed")
    plt.imshow(fixed_slice, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("Moving (registered)")
    plt.imshow(moving_slice, cmap="gray")
    plt.show()

def show_overlap_slices(fixed, moving):
    fixed = itk_to_numpy(fixed)
    moving = itk_to_numpy(moving)
    z = fixed.shape[0] // 2
    plt.figure(figsize=(8, 8))
    plt.imshow(fixed[z], cmap="gray")
    plt.imshow(moving[z], cmap="Reds", alpha=0.35)
    plt.title("Overlap")
    plt.show()

def show_difference(fixed, moving):
    fixed = itk_to_numpy(fixed)
    moving = itk_to_numpy(moving)
    z = fixed.shape[0] // 2
    diff = np.abs(fixed - moving)
    plt.figure(figsize=(8, 8))
    plt.imshow(diff[z], cmap="hot")
    plt.colorbar()
    plt.title("Difference")
    plt.show()
