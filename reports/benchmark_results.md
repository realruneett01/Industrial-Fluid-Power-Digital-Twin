# Benchmark Evaluation Report

## Temperature-shift generalization

| pipeline | target | f1_in | f1_shift | f1_shift_all_labels | delta_f1 |
| --- | --- | --- | --- | --- | --- |
| statistical | cooler | 0.6579 | 0.2640 | 0.1760 | -0.3939 |
| statistical | valve | 0.9991 | 0.7853 | 0.7853 | -0.2138 |
| statistical | pump | 1.0000 | 0.6110 | 0.6110 | -0.3890 |
| statistical | accumulator | 0.2279 | 0.1303 | 0.1303 | -0.0976 |
| physics | cooler | 0.6533 | 0.2717 | 0.1811 | -0.3816 |
| physics | valve | 0.9725 | 0.7228 | 0.7228 | -0.2497 |
| physics | pump | 0.7937 | 0.4607 | 0.4607 | -0.3330 |
| physics | accumulator | 0.3970 | 0.2815 | 0.2815 | -0.1155 |
| hybrid | cooler | 0.6579 | 0.2624 | 0.1749 | -0.3955 |
| hybrid | valve | 1.0000 | 0.7228 | 0.7228 | -0.2772 |
| hybrid | pump | 1.0000 | 0.6110 | 0.6110 | -0.3890 |
| hybrid | accumulator | 0.3436 | 0.2741 | 0.2741 | -0.0695 |
| cnn_1d | cooler | 0.6567 | 0.2670 | 0.1780 | -0.3897 |
| cnn_1d | valve | 0.9871 | 0.5005 | 0.5005 | -0.4865 |
| cnn_1d | pump | 0.9644 | 0.5101 | 0.5101 | -0.4543 |
| cnn_1d | accumulator | 0.1463 | 0.0814 | 0.0814 | -0.0649 |

## Four-way comparison - blocked 5-fold

| pipeline | target | macro_f1 | macro_f1_all_labels | balanced_accuracy | fit_seconds |
| --- | --- | --- | --- | --- | --- |
| statistical | cooler | 0.8487 | 0.8487 | 0.8507 | 21.3618 |
| statistical | valve | 0.9700 | 0.9700 | 0.9707 | 22.8830 |
| statistical | pump | 0.8854 | 0.8854 | 0.8895 | 20.8299 |
| statistical | accumulator | 0.3149 | 0.3149 | 0.3258 | 23.1180 |
| physics | cooler | 0.8185 | 0.8185 | 0.8293 | 16.6824 |
| physics | valve | 0.8462 | 0.8462 | 0.8815 | 16.9406 |
| physics | pump | 0.8112 | 0.8112 | 0.8002 | 17.0056 |
| physics | accumulator | 0.4221 | 0.4221 | 0.4373 | 18.1276 |
| hybrid | cooler | 0.8792 | 0.8792 | 0.8821 | 22.1368 |
| hybrid | valve | 0.9764 | 0.9764 | 0.9771 | 23.2573 |
| hybrid | pump | 0.9089 | 0.9089 | 0.9126 | 21.9153 |
| hybrid | accumulator | 0.3253 | 0.3253 | 0.3215 | 23.9561 |
| cnn_1d | cooler | 0.8756 | 0.8756 | 0.8785 | 9.1818 |
| cnn_1d | valve | 0.9972 | 0.9972 | 0.9963 | 9.1818 |
| cnn_1d | pump | 0.8856 | 0.8856 | 0.8858 | 9.1818 |
| cnn_1d | accumulator | 0.2127 | 0.2127 | 0.2103 | 9.1818 |

## Experimental Notes and Audit

- **Class coverage audit (blocked folds)**: {'cooler': False, 'valve': True, 'pump': True, 'accumulator': False}
- **Temperature threshold (TS1 cycle mean)**: 44.84 degC
