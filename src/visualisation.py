import itk
import numpy as np
import vtk


def _to_itk_with_meta(arr, reference):
    """ITK image with spacing, origin and direction"""
    img = itk.GetImageFromArray(arr.astype(np.uint8))
    img.SetSpacing(reference.GetSpacing())
    img.SetOrigin(reference.GetOrigin())
    img.SetDirection(reference.GetDirection())
    return img


def compute_changes(mask1, mask2):
    """Return what's been add, remove and stable"""
    m1 = itk.GetArrayFromImage(mask1) > 0
    m2 = itk.GetArrayFromImage(mask2) > 0

    added = m2 & ~m1
    removed = m1 & ~m2
    stable = m1 & m2

    added_img = _to_itk_with_meta(added, mask1)
    removed_img = _to_itk_with_meta(removed, mask1)
    stable_img = _to_itk_with_meta(stable, mask1)

    return added_img, removed_img, stable_img


def compute_metrics(mask1, mask2, img1, img2):
    """Volume, voxel intensity and creative (dice)"""
    vol1 = int((itk.GetArrayFromImage(mask1) > 0).sum())
    vol2 = int((itk.GetArrayFromImage(mask2) > 0).sum())
    delta_vol = vol2 - vol1

    img1_np = itk.GetArrayFromImage(img1)
    img2_np = itk.GetArrayFromImage(img2)
    m1_np = itk.GetArrayFromImage(mask1) > 0
    m2_np = itk.GetArrayFromImage(mask2) > 0

    mean1 = float(img1_np[m1_np].mean()) if m1_np.any() else 0.0
    mean2 = float(img2_np[m2_np].mean()) if m2_np.any() else 0.0

    dice = 2 * (m1_np & m2_np).sum() / (m1_np.sum() + m2_np.sum() + 1e-8)

    return {
        "vol1": vol1,
        "vol2": vol2,
        "delta_vol": delta_vol,
        "mean1": mean1,
        "mean2": mean2,
        "dice": float(dice),
    }


def _make_lut(r, g, b):
    """LUT: 0 = transparent, 1 = RGB"""
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(2)
    lut.SetTableValue(0, 0, 0, 0, 0.0)
    lut.SetTableValue(1, r, g, b, 1.0)
    lut.Build()
    return lut


def render_change_overlay(base, added, removed, stable, metrics):
    """
    Params:
        base: itk image, initial state
        added, removed, stable: difference with initial state
        metrics : dict with vol1, vol2, delta_vol, mean1, mean2, dice
    """
    # itk to vtk
    base_vtk = itk.vtk_image_from_image(base)
    add_vtk = itk.vtk_image_from_image(added)
    rem_vtk = itk.vtk_image_from_image(removed)
    st_vtk = itk.vtk_image_from_image(stable)

    base_arr = itk.GetArrayFromImage(base).astype(float)
    w = float(base_arr.max() - base_arr.min())
    l = float(base_arr.min() + w / 2.0)

    # itk to ndarray
    stable_arr = itk.GetArrayFromImage(stable) > 0
    added_arr = itk.GetArrayFromImage(added) > 0
    removed_arr = itk.GetArrayFromImage(removed) > 0
    tumor_union = stable_arr | added_arr | removed_arr

    slice_sums = tumor_union.sum(axis=(0, 1))
    best_slice = (
        int(np.argmax(slice_sums)) if slice_sums.max() > 0 else base_arr.shape[2] // 2
    )

    # VTK render
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.1, 0.1, 0.1)

    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(800, 800)
    render_window.SetWindowName("Visualization Toolkit - Tumor Change Overlay")
    render_window.AddRenderer(renderer)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    mappers = []
    dims = base_vtk.GetDimensions()
    current_axis = ["x"]
    current_slice = [best_slice]

    orientation_setters = {
        "x": ("SetOrientationToX", 0),
        "y": ("SetOrientationToY", 1),
        "z": ("SetOrientationToZ", 2),
    }

    def add_layer(img, opacity, lut=None, window=None, level=None):
        """Useful to render original CT with multiple overlay"""
        mapper = vtk.vtkImageSliceMapper()
        mapper.SetInputData(img)
        mapper.SetOrientationToX()
        mapper.SetSliceNumber(current_slice[0])
        mappers.append(mapper)

        actor = vtk.vtkImageSlice()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(opacity)

        if lut is not None:
            actor.GetProperty().SetLookupTable(lut)
            actor.GetProperty().UseLookupTableScalarRangeOn()
        elif window is not None and level is not None:
            actor.GetProperty().SetColorWindow(window)
            actor.GetProperty().SetColorLevel(level)

        renderer.AddActor(actor)

    add_layer(base_vtk, 1.0, window=w, level=l)
    add_layer(st_vtk, 0.5, lut=_make_lut(1, 1, 0))
    add_layer(add_vtk, 0.8, lut=_make_lut(0, 1, 0))
    add_layer(rem_vtk, 0.8, lut=_make_lut(1, 0, 0))

    # camera settings
    camera = renderer.GetActiveCamera()
    camera.SetPosition(1, 0, 0)
    camera.SetFocalPoint(0, 0, 0)
    camera.SetViewUp(0, 0, 1)
    camera.ParallelProjectionOn()
    renderer.ResetCamera()

    # Overlay metrics
    delta_pct = (
        (metrics["delta_vol"] / metrics["vol1"] * 100) if metrics["vol1"] > 0 else 0.0
    )
    delta_intensity = metrics["mean2"] - metrics["mean1"]

    metric_modes = ["volume", "intensity", "dice"]
    current_mode = [0]

    def format_metrics(mode):
        if mode == "volume":
            return (
                f"VOLUME\n"
                f"Init : {metrics['vol1']} vox\n"
                f"New : {metrics['vol2']} vox\n"
                f"Delta : {metrics['delta_vol']:+d} vox ({delta_pct:+.1f}%)"
            )
        if mode == "intensity":
            return (
                f"INTENSITE\n"
                f"Avg Init : {metrics['mean1']:.1f}\n"
                f"Avg New : {metrics['mean2']:.1f}\n"
                f"Delta : {delta_intensity:+.1f}"
            )
        return f"CHEVAUCHEMENT\nDice : {metrics['dice']:.3f}"

    # top left - metrics
    text_actor = vtk.vtkTextActor()
    text_actor.SetInput(format_metrics(metric_modes[current_mode[0]]))
    tprop = text_actor.GetTextProperty()
    tprop.SetFontSize(18)
    tprop.SetColor(1, 1, 1)
    tprop.SetFontFamilyToArial()
    tprop.SetJustificationToRight()
    tprop.SetVerticalJustificationToTop()
    text_actor.SetTextScaleModeToNone()
    text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
    text_actor.GetPositionCoordinate().SetValue(0.98, 0.97)
    renderer.AddActor2D(text_actor)

    # bottom right - hint
    hint_actor = vtk.vtkTextActor()
    hint_actor.SetInput(
        "[Click Droit] changer de metrique\n[Click Gauche] changer d'axe\n[Scroll] changer de slice"
    )
    htprop = hint_actor.GetTextProperty()
    htprop.SetFontSize(12)
    htprop.SetColor(0.7, 0.7, 0.7)
    htprop.SetJustificationToRight()
    htprop.SetVerticalJustificationToBottom()
    hint_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
    hint_actor.GetPositionCoordinate().SetValue(0.98, 0.02)
    renderer.AddActor2D(hint_actor)

    def set_axis(axis):
        """Set axis: Coronal, Sagittal and Transverse"""
        if axis == current_axis[0]:
            return
        setter_name, dim_idx = orientation_setters[axis]
        current_axis[0] = axis
        current_slice[0] = dims[dim_idx] // 2
        for m in mappers:
            getattr(m, setter_name)()
            m.SetSliceNumber(current_slice[0])
        render_window.Render()

    def axis_from_camera():
        dx, dy, dz = camera.GetDirectionOfProjection()
        ax, ay, az = abs(dx), abs(dy), abs(dz)
        if ax >= ay and ax >= az:
            return "x"
        if ay >= ax and ay >= az:
            return "y"
        return "z"

    class SliceAndRotateStyle(vtk.vtkInteractorStyleTrackballCamera):
        """Overwrite style to switch between slices, angles and metrics"""
        def __init__(self):
            self.AddObserver("MouseWheelForwardEvent", self._scroll_forward)
            self.AddObserver("MouseWheelBackwardEvent", self._scroll_backward)
            self.AddObserver("LeftButtonReleaseEvent", self._on_rotation_end)
            self.AddObserver("RightButtonReleaseEvent", self._switch_metrics)

        def _scroll_forward(self, obj, event):
            _, dim_idx = orientation_setters[current_axis[0]]
            current_slice[0] = min(current_slice[0] + 1, dims[dim_idx] - 1)
            self._update_slice()

        def _scroll_backward(self, obj, event):
            current_slice[0] = max(current_slice[0] - 1, 0)
            self._update_slice()

        def _update_slice(self):
            for m in mappers:
                m.SetSliceNumber(current_slice[0])
            render_window.Render()
            print(f"\rPlan {current_axis[0].upper()} - coupe {current_slice[0]}", end="", flush=True)

        def _on_rotation_end(self, obj, event):
            new_axis = axis_from_camera()
            set_axis(new_axis)

        def _switch_metrics(self, obj, event):
            current_mode[0] = (current_mode[0] + 1) % len(metric_modes)
            text_actor.SetInput(format_metrics(metric_modes[current_mode[0]]))
            render_window.Render()

    style = SliceAndRotateStyle()
    interactor.SetInteractorStyle(style)

    render_window.Render()
    interactor.Start()
