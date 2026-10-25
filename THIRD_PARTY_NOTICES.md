# Third-party components

These components are **not** covered by the proprietary license in `LICENSE`. Each stays under its
own license; the Owner has no right to relicense them. Versions are whatever `pip` resolves; the
license names below were read from package metadata on 2026-10-25 and should be re-checked for the
exact versions you ship.

| Component | License | Home |
| --- | --- | --- |
| numpy | BSD-3-Clause (plus bundled permissive components) | https://numpy.org |
| scipy | BSD-3-Clause | https://scipy.org |
| pandas | BSD-3-Clause | https://pandas.pydata.org |
| PyYAML | MIT | https://pyyaml.org |
| onnxruntime | MIT | https://onnxruntime.ai |
| asyncua | LGPL-3.0-or-later (see note 1) | https://github.com/FreeOpcUa/opcua-asyncio |
| cryptography | Apache-2.0 OR BSD-3-Clause | https://cryptography.io |
| scikit-learn | BSD-3-Clause (training only) | https://scikit-learn.org |
| CatBoost | Apache-2.0 (training only) | https://catboost.ai |
| PyTorch | BSD-3-Clause and others (training only) | https://pytorch.org |
| onnx | Apache-2.0 (training only) | https://onnx.ai |
| pyarrow | Apache-2.0 (training only) | https://arrow.apache.org |

## Notes

1. **asyncua is LGPL-3.0-or-later.** hydro only imports it as a separate library; it is not modified or
   copied into hydro's files. When you ship a wheelhouse or container that contains asyncua you must
   also ship the LGPL text and must not stop the recipient from replacing asyncua with a modified
   version of it (this is why `scripts/build_portable.py` ships wheels, not a frozen binary).
2. **Apache-2.0 and BSD components** require their license text and copyright notices to accompany
   redistributed copies. `pip download` wheels include them in their `*.dist-info` folders; do not
   strip those folders from a wheelhouse.

## Dataset

The ZeMA hydraulic test-rig data (UCI Machine Learning Repository dataset 447, "Condition monitoring
of hydraulic systems") is **not** part of this repository and is **not** owned by the Owner. It is
published by its authors under CC BY 4.0, which requires attribution. Cite:

> N. Helwig, E. Pignanelli, A. Schuetze, "Condition Monitoring of a Complex Hydraulic System Using
> Multivariate Statistics", Proc. I2MTC 2015, IEEE, doi:10.1109/I2MTC.2015.7151267.

Models trained on it are derived from CC BY 4.0 data; keep the attribution with any model bundle you
hand to a customer. Verify the current license terms on the dataset page before commercial use.

## Generated media

The scene images in `docs/assets/explainer/` were produced with a generative image model. The
copyright status of purely machine-generated images differs between countries (in several it is
unprotected), so the proprietary license may not be enforceable for those files.
