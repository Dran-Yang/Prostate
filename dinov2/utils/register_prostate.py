"""
Rigid + affine registration helper for prostate MRI.

Usage:
  python -m dinov2.utils.register_prostate ^
      --data-root <PATH_TO_DATA_ROOT> ^
      --output-root <PATH_TO_SAVE_REGISTERED> ^
      --ref-seq ax_t2wi ^
      --moving-seqs ax_adc ax_dwi ^
      --seg-name roi_Prostate

Notes:
- Requires `pip install SimpleITK`.
- Picks the highest-b DWI file matching prefix (e.g., ax_dwi_1500.nii vs ax_dwi_1000.nii).
- Resamples masks with nearest-neighbor; images with linear.
"""

import argparse
import re
from pathlib import Path
from typing import Iterable, Sequence

import SimpleITK as sitk


def _resolve_fp(folder: Path, stem: str) -> Path:
    for ext in (".nii", ".nii.gz"):
        cand = folder / f"{stem}{ext}"
        if cand.exists():
            return cand
    return folder / f"{stem}.nii"


def _pick_dwi(folder: Path, prefixes: Sequence[str]) -> Path:
    candidates: list[Path] = []
    for prefix in prefixes:
        candidates.extend(folder.glob(f"{prefix}*.nii*"))
    if not candidates:
        return _resolve_fp(folder, "ax_dwi")

    def _score(p: Path) -> int:
        m = re.search(r"(\d+)", p.stem)
        return int(m.group(1)) if m else -1

    candidates.sort(key=_score, reverse=True)
    return candidates[0]


def _register_single(
    fixed_fp: Path, moving_fp: Path, mode: str = "affine"
) -> sitk.Transform:
    fixed = sitk.ReadImage(str(fixed_fp))
    moving = sitk.ReadImage(str(moving_fp))

    initial = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.1)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=200,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(initial, inPlace=False)
    reg.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    if mode == "rigid":
        transform = sitk.Euler3DTransform()
    else:
        transform = sitk.AffineTransform(3)

    reg.SetInitialTransform(transform, inPlace=False)
    final_transform = reg.Execute(fixed, moving)
    return final_transform


def _resample(moving_fp: Path, ref_fp: Path, transform: sitk.Transform, is_mask: bool) -> sitk.Image:
    moving = sitk.ReadImage(str(moving_fp))
    ref = sitk.ReadImage(str(ref_fp))
    return sitk.Resample(
        moving,
        ref,
        transform,
        sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear,
        0,
        moving.GetPixelID(),
    )


def register_patient(
    patient_dir: Path,
    output_dir: Path,
    ref_seq: str,
    moving_seqs: Iterable[str],
    seg_name: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    ref_fp = _resolve_fp(patient_dir, ref_seq)

    # pick dwi file dynamically
    seq_to_fp = {seq: _resolve_fp(patient_dir, seq) for seq in moving_seqs}
    if "ax_dwi" in seq_to_fp and not seq_to_fp["ax_dwi"].exists():
        seq_to_fp["ax_dwi"] = _pick_dwi(patient_dir, prefixes=("ax_dwi", "dwi"))

    transform = None
    for seq, moving_fp in seq_to_fp.items():
        if not moving_fp.exists():
            print(f"[warn] {seq} missing in {patient_dir}")
            continue
        if transform is None:
            transform = _register_single(ref_fp, moving_fp, mode="affine")
        registered = _resample(moving_fp, ref_fp, transform, is_mask=False)
        sitk.WriteImage(registered, str(output_dir / moving_fp.name))

    seg_fp = _resolve_fp(patient_dir, seg_name)
    if seg_fp.exists() and transform is not None:
        seg_resampled = _resample(seg_fp, ref_fp, transform, is_mask=True)
        sitk.WriteImage(seg_resampled, str(output_dir / seg_fp.name))

    # always copy reference to output
    sitk.WriteImage(sitk.ReadImage(str(ref_fp)), str(output_dir / ref_fp.name))


def main():
    parser = argparse.ArgumentParser(description="Rigid+affine registration for prostate MRI.")
    parser.add_argument("--data-root", required=True, help="Root containing patient folders.")
    parser.add_argument(
        "--output-root",
        required=True,
        help="Where to write registered volumes (mirrors patient folder names).",
    )
    parser.add_argument("--ref-seq", default="ax_t2wi", help="Reference sequence stem.")
    parser.add_argument(
        "--moving-seqs",
        nargs="+",
        default=["ax_adc", "ax_dwi"],
        help="Moving sequence stems to align to reference.",
    )
    parser.add_argument("--seg-name", default="roi_Prostate", help="Segmentation stem.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.output_root)
    for patient_dir in data_root.iterdir():
        if not patient_dir.is_dir():
            continue
        register_patient(
            patient_dir,
            out_root / patient_dir.name,
            ref_seq=args.ref_seq,
            moving_seqs=args.moving_seqs,
            seg_name=args.seg_name,
        )


if __name__ == "__main__":
    main()
