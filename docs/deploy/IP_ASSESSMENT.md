# What is actually proprietary here

Confidential. This is an engineering assessment of the repository, not legal advice. Whether something is patentable, copyrightable or a trade secret is a legal question for a qualified IP attorney in the relevant countries.

## Built on public or standard components (not ownable)

- The ZeMA/UCI 447 dataset (public, CC BY 4.0) and the way it is split for blocked cross-validation.
- LDA, Random Forest, CatBoost, 1D-CNN architectures and ONNX export.
- The Walther viscosity equation (ASTM D341), OPC-UA, ERPNext's REST API.
- Health Index as the mean healthy-class probability (a simple averaging rule).

## What is original to this project (candidate protectable assets)

1. **Source code, as a whole.** Automatic copyright protection in most countries, with no registration needed to exist (registration helps enforcement in some, for example the US).
2. **The integrated grey-box feature set**: temperature-normalised pump-leakage coefficient, lumped hydraulic capacitance during pressure build-up, valve-edge delay with a fold-fitted healthy baseline, and a causal multi-cycle thermal-effectiveness window. Individual ideas are textbook physics; the specific combination, parameterisation and leakage-free fitting procedure is the original contribution. Best protected as a trade secret and as code.
3. **The read-only OPC-UA boundary**: a user manager that never grants Admin (because asyncua's Admin role bypasses node access levels), now paired with a runtime self-test that attempts a write and fails the deployment if it succeeds.
4. **The deployment gate**: portable drive that checks its own integrity, discovers sources, accepts or rejects sensors channel by channel, measures latency against the SLA on the target machine, attests the read-only boundary, and refuses to start on any failure ("fail closed"), with machine-based auto-configuration.
5. **Stream robustness logic**: cycle assembly with gap/restart/duplicate handling and matching reset of the predictor's rolling window.
6. **Trained model bundles**, once trained on proprietary plant data (the demo bundle trained on the public dataset carries CC BY attribution duties).

## What the results do not support claiming

- Adding the physics features to CatBoost helped only modestly (hybrid beats statistical-only on all four targets, by 0.01-0.03 macro-F1), the hypothesis that they resist temperature shift was not confirmed, and the plain LDA baseline in `reports/baseline_results.md` scores higher than every CatBoost/CNN pipeline on all four targets. There is no demonstrated accuracy advantage to market.
- The deployed CatBoost bundle is weak on the accumulator target (macro-F1 about 0.33 versus 0.80 for LDA).
- Nothing has been validated on a field installation.

## Realistic protection strategy

1. Keep the repository private, restrict access to named people, and use NDAs: trade-secret protection requires reasonable secrecy measures.
2. Use the proprietary license in `LICENSE` and a signed agreement for any recipient.
3. Do not publish benchmark numbers or the feature definitions.
4. If a patent is wanted, speak to a patent attorney **before** any public disclosure; features 3 and 4 above are the likeliest candidates, and patentability is uncertain for software methods in many countries.
5. The earlier AGPL commits remain licensed to anyone who obtained those copies. Relicensing covers new versions only, and only the copyright holder(s) can do it: if anyone else contributed code, get their written consent.
