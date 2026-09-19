# LightGBM results

## Diabetes test-set class distribution

The LightGBM diabetes evaluation uses 821 test instances with the following
target distribution:

| Class | Target outcome | Instances | Percentage |
|---|---|---:|---:|
| Negative | No hypoglycemic event tomorrow | 720 | 87.70% |
| Positive | Hypoglycemic event tomorrow | 101 | 12.30% |
| **Total** | | **821** | **100.00%** |

The negative-to-positive ratio is approximately **7.13:1**. Consequently, an
always-negative classifier would obtain **87.70% accuracy** while detecting
none of the positive cases. Accuracy should therefore be interpreted together
with positive-class recall and macro F1.
