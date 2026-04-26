import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk


# ============================================================
# Synthetic image generation
# ============================================================

def make_test_image(size=(180, 180)):
    yy, xx = np.mgrid[0:size[0], 0:size[1]]
    img = np.zeros(size, dtype=np.float32)

    blobs = [
        (45, 50, 10, 1.0),
        (120, 60, 18, 0.8),
        (80, 130, 12, 0.9),
        (130, 130, 14, 0.7),
    ]
    for cy, cx, s, a in blobs:
        img += a * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s ** 2))

    img[20:40, 110:145] += 0.5
    img[95:140, 25:40] += 0.35

    img -= img.min()
    img /= img.max()
    return img.astype(np.float32)


def to_sitk(img_np):
    img = sitk.GetImageFromArray(img_np)
    img.SetSpacing((1.0, 1.0))
    img.SetOrigin((0.0, 0.0))
    return img


# ============================================================
# Image utilities
# ============================================================

def resample_with_transform(moving, reference, transform, default_value=0.0):
    return sitk.Resample(
        moving,
        reference,
        transform,
        sitk.sitkLinear,
        default_value,
        moving.GetPixelID()
    )


def image_to_np(img):
    return sitk.GetArrayFromImage(img)


def make_overlay(fixed_np, moving_np):
    rgb = np.zeros(fixed_np.shape + (3,), dtype=np.float32)
    rgb[..., 0] = fixed_np
    rgb[..., 1] = moving_np
    return np.clip(rgb, 0.0, 1.0)


# ============================================================
# Registration
# ============================================================

def build_registration_method(metric_name):
    R = sitk.ImageRegistrationMethod()

    # Interpolator
    R.SetInterpolator(sitk.sitkLinear)

    # Metric
    if metric_name == "MeanSquares":
        R.SetMetricAsMeanSquares()
    elif metric_name == "Correlation":
        R.SetMetricAsCorrelation()
    elif metric_name == "JointHistogramMI" :
        R.SetMetricAsJointHistogramMutualInformation(numberOfHistogramBins=100)
    elif metric_name == "MattesMI":
        R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100)
    else:
        raise ValueError(f"Unknown metric: {metric_name}")

    # Use a subset of points for speed/reproducibility
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.25, seed=42)

    # Multi-resolution
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Optimizer
    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-3,
        numberOfIterations=100,
        relaxationFactor=0.5,
        gradientMagnitudeTolerance=1e-6
    )
    R.SetOptimizerScalesFromPhysicalShift()

    return R


def run_registration(fixed, moving, metric_name):
    fixed_f = sitk.Cast(fixed, sitk.sitkFloat32)
    moving_f = sitk.Cast(moving, sitk.sitkFloat32)

    initial_transform = sitk.TranslationTransform(fixed.GetDimension())
    initial_transform.SetOffset((0.0, 0.0))

    R = build_registration_method(metric_name)
    R.SetInitialTransform(initial_transform, inPlace=False)

    history = {
        "iteration": [],
        "metric": [],
        "tx": [],
        "ty": [],
        "level": [],
    }

    current_transform_holder = {"transform": None}

    def iteration_callback():
        # TranslationTransform parameters are [tx, ty]
        params = list(R.GetOptimizerPosition())

        history["iteration"].append(R.GetOptimizerIteration())
        history["metric"].append(R.GetMetricValue())
        history["tx"].append(params[0])
        history["ty"].append(params[1])
        history["level"].append(R.GetCurrentLevel())

    def start_callback():
        history["iteration"].clear()
        history["metric"].clear()
        history["tx"].clear()
        history["ty"].clear()
        history["level"].clear()

    R.AddCommand(sitk.sitkStartEvent, start_callback)
    R.AddCommand(sitk.sitkIterationEvent, iteration_callback)

    final_transform = R.Execute(fixed_f, moving_f)
    current_transform_holder["transform"] = final_transform

    registered = resample_with_transform(moving_f, fixed_f, final_transform)

    result = {
        "metric_name": metric_name,
        "history": history,
        "final_transform": final_transform,
        "registered": registered,
        "stop_condition": R.GetOptimizerStopConditionDescription(),
        "final_metric": R.GetMetricValue(),
    }
    return result


# ============================================================
# Demo data creation
# ============================================================

def create_demo_pair():
    fixed_np = make_test_image()
    fixed = to_sitk(fixed_np)

    # Create moving by applying a known translation
    true_shift = (6.0, -4.0)  # tx, ty
    create_transform = sitk.TranslationTransform(2)
    create_transform.SetOffset(true_shift)

    moving = resample_with_transform(fixed, fixed, create_transform)

    # Intensity modification to make metric differences visible
    moving = sitk.Cast(moving, sitk.sitkFloat32)
    moving = moving * 1.3 + 0.10

    # Add noise
    rng = np.random.default_rng(7)
    moving_np = sitk.GetArrayFromImage(moving)
    moving_np = np.clip(moving_np + 0.03 * rng.standard_normal(moving_np.shape), 0, 1).astype(np.float32)
    moving = to_sitk(moving_np)

    # To align moving back to fixed, the ideal translation is approximately -true_shift
    true_correction = (-true_shift[0], -true_shift[1])

    return fixed, moving, true_correction


# ============================================================
# Plotting
# ============================================================

def plot_initial_and_final(fixed, moving, results):
    fixed_np = image_to_np(fixed)
    moving_np = image_to_np(moving)

    fig, axes = plt.subplots(len(results), 3, figsize=(12, 4 * len(results)), constrained_layout=True)
    if len(results) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, res in enumerate(results):
        reg_np = image_to_np(res["registered"])

        axes[row, 0].imshow(make_overlay(fixed_np, moving_np))
        axes[row, 0].set_title(f"{res['metric_name']} - initial overlay")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(make_overlay(fixed_np, reg_np))
        axes[row, 1].set_title(f"{res['metric_name']} - final overlay")
        axes[row, 1].axis("off")

        diff = fixed_np - reg_np
        im = axes[row, 2].imshow(diff, cmap="gray")
        axes[row, 2].set_title(f"{res['metric_name']} - fixed - registered")
        axes[row, 2].axis("off")
        plt.colorbar(im, ax=axes[row, 2], fraction=0.046, pad=0.04)

    return fig


def plot_convergence(results, true_correction):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4), constrained_layout=True)
    if len(results) == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        h = res["history"]
        it = np.array(h["iteration"], dtype=float)
        metric = np.array(h["metric"], dtype=float)
        tx = np.array(h["tx"], dtype=float)
        ty = np.array(h["ty"], dtype=float)

        err = np.sqrt((tx - true_correction[0]) ** 2 + (ty - true_correction[1]) ** 2)

        ax2 = ax.twinx()
        ax.plot(it, metric, "o-", label="metric")
        ax2.plot(it, err, "s--", label="parameter error")

        ax.set_title(res["metric_name"])
        ax.set_xlabel("iteration")
        ax.set_ylabel("metric value")
        ax2.set_ylabel("distance to true shift")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)

    return fig


def plot_parameter_trajectory(results, true_correction):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4), constrained_layout=True)
    if len(results) == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        h = res["history"]
        tx = np.array(h["tx"], dtype=float)
        ty = np.array(h["ty"], dtype=float)

        ax.plot(tx, ty, "o-", label="optimizer path")
        ax.plot(true_correction[0], true_correction[1], "rx", ms=10, mew=2, label="true correction")
        ax.set_title(res["metric_name"])
        ax.set_xlabel("tx")
        ax.set_ylabel("ty")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        ax.legend()

    return fig


# ============================================================
# Main
# ============================================================

def main():
    fixed, moving, true_correction = create_demo_pair()

    metric_names = ["MeanSquares", "Correlation", "MattesMI"]
    results = []

    for metric_name in metric_names:
        res = run_registration(fixed, moving, metric_name)
        results.append(res)

    print("True correction (tx, ty):", true_correction)
    print()

    for res in results:
        final_params = list(res["final_transform"].GetParameters())
        print(f"{res['metric_name']}:")
        print(f"  final translation = ({final_params[0]:.3f}, {final_params[1]:.3f})")
        print(f"  final metric      = {res['final_metric']:.6f}")
        print(f"  stop condition    = {res['stop_condition']}")
        print()

    plot_initial_and_final(fixed, moving, results)
    plot_convergence(results, true_correction)
    plot_parameter_trajectory(results, true_correction)

    plt.show()


if __name__ == "__main__":
    main()