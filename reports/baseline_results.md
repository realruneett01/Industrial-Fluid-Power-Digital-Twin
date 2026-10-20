# Benchmark Evaluation Report

## Baseline (LDA, RF) - blocked 5-fold

| pipeline | target | macro_f1 | macro_f1_all_labels | balanced_accuracy | fit_seconds |
| --- | --- | --- | --- | --- | --- |
| lda | cooler | 0.9823 | 0.9823 | 0.9823 | 0.7975 |
| lda | valve | 0.9977 | 0.9977 | 0.9977 | 0.7277 |
| lda | pump | 0.9915 | 0.9915 | 0.9917 | 0.7304 |
| lda | accumulator | 0.8019 | 0.8019 | 0.7981 | 0.7233 |
| rf | cooler | 0.7880 | 0.7880 | 0.7956 | 2.5896 |
| rf | valve | 0.9324 | 0.9324 | 0.9276 | 2.3430 |
| rf | pump | 0.8955 | 0.8955 | 0.8977 | 2.2938 |
| rf | accumulator | 0.1426 | 0.1426 | 0.1625 | 2.5306 |

## Experimental Notes and Audit

- **Class coverage audit (blocked folds)**: {'cooler': False, 'valve': True, 'pump': True, 'accumulator': False}
