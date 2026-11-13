# kaiLLM throughput

All results below were obtained with kaiLLM 0.1.0rc4, without speculative decoding unless the row says otherwise.

Generation uses 512 input tokens and 256 generated tokens, including prefill. Prefill uses 512 or 2048 input tokens. Rates are aggregate tokens/s. "Weights in VRAM" is the memory held by the model's weights after loading, compared with the GGUF file size.

**These numbers aren't realistic for real use cases.** Better benchmarks are planned.

## RTX 3060

| Checkpoint | Batch 1 | Batch 2 | Batch 8 | Batch 16 | Prefill 512 | Prefill 2048 |
|---|---:|---:|---:|---:|---:|---:|
| chimachima 500 MB | 147.8 | 242.5 | 584.3 | 819.9 | 3,494 | 4,273 |
| chimachima 750 MB | 155.8 | 235.8 | 587.4 | 799.8 | 3,532 | 4,235 |
| chimachima 1000 MB | 166.0 | 242.5 | 585.3 | 806.0 | 3,477 | 4,246 |
| chimachima 1250 MB | 169.4 | 280.3 | 617.3 | 1,013 | 5,220 | 6,073 |
| chimachima 1500 MB | 167.5 | 298.7 | 610.5 | 1,175 | 7,155 | 7,824 |
| suisui Q4_0_XXS | 223.9 | 402.3 | 780.9 | 1,407 | 8,038 | 8,836 |
| suisui Q4_0_XS | 196.9 | 353.2 | 783.0 | 1,320 | 8,235 | 8,970 |
| suisui Q8_0_S | 168.6 | 315.8 | 764.2 | 1,245 | 8,120 | 8,914 |
| AtomicChat AD-IQ3_S | 210.5 | 372.4 | 764.8 | 1,356 | 7,824 | 8,574 |
| LiquidAI QAD-Q4_0 | 220.5 | 409.0 | 806.1 | 1,417 | 8,123 | 8,923 |
| AtomicChat AD-Q5_K_M | 188.7 | 338.2 | 647.5 | 1,273 | 7,345 | 8,081 |
| AtomicChat AD-Q6_K | 163.6 | 299.4 | 647.3 | 1,171 | 7,338 | 8,040 |
| LiquidAI Q8_0 | 146.7 | 277.8 | 732.3 | 1,153 | 8,015 | 8,815 |

## A40

| Checkpoint | Batch 1 | Batch 2 | Batch 8 | Batch 16 | Prefill 512 | Prefill 2048 | Weights in VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| chimachima 500 MB | 233.0 | 311.5 | 1,040 | 1,607 | 7,244 | 11,021 | 472 MiB (1.002x file) |
| chimachima 750 MB | 221.7 | 312.7 | 1,047 | 1,568 | 7,202 | 10,983 | 705 MiB (1.001x file) |
| chimachima 1000 MB | 241.1 | 316.9 | 1,039 | 1,553 | 7,218 | 11,087 | 927 MiB (1.001x file) |
| chimachima 1250 MB | 253.4 | 405.9 | 1,193 | 1,775 | 10,064 | 14,158 | 1,182 MiB (0.997x file) |
| chimachima 1500 MB | 261.6 | 467.3 | 1,229 | 1,838 | 12,087 | 16,165 | 1,418 MiB (0.996x file) |
| suisui Q4_0_XXS | 321.2 | 596.7 | 1,467 | 2,198 | 12,737 | 17,499 | 948 MiB (0.994x file) |
| suisui Q4_0_XS | 296.7 | 533.5 | 1,449 | 2,069 | 13,304 | 17,961 | 1,186 MiB (0.995x file) |
| suisui Q8_0_S | 261.1 | 475.9 | 1,383 | 1,948 | 12,967 | 17,564 | 1,425 MiB (0.996x file) |
| AtomicChat AD-IQ3_S | 331.2 | 595.3 | 1,514 | 2,199 | 14,666 | 18,714 | 873 MiB (0.993x file) |
| LiquidAI QAD-Q4_0 | 330.3 | 611.0 | 1,534 | 2,162 | 13,639 | 18,234 | 995 MiB (0.994x file) |
| AtomicChat AD-Q5_K_M | 286.2 | 519.7 | 1,319 | 1,986 | 13,608 | 17,170 | 1,244 MiB (0.995x file) |
| AtomicChat AD-Q6_K | 247.7 | 465.1 | 1,286 | 1,832 | 12,700 | 17,433 | 1,509 MiB (0.996x file) |
| LiquidAI Q8_0 | 225.7 | 427.1 | 1,310 | 1,812 | 13,501 | 17,917 | 1,755 MiB (0.997x file) |

## RTX 5090

| Checkpoint | Batch 1 | Batch 2 | Batch 8 | Batch 16 | Prefill 512 | Prefill 2048 | Weights in VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| chimachima 500 MB | 461.3 | 635.9 | 2,051 | 2,872 | 17,005 | 30,814 | 472 MiB (1.003x file) |
| chimachima 750 MB | 426.3 | 575.5 | 1,926 | 2,844 | 16,603 | 30,381 | 706 MiB (1.002x file) |
| chimachima 1000 MB | 475.9 | 631.3 | 1,971 | 2,926 | 19,922 | 30,764 | 927 MiB (1.001x file) |
| chimachima 1250 MB | 446.5 | 764.7 | 2,472 | 3,550 | 21,865 | 39,675 | 1,182 MiB (0.997x file) |
| chimachima 1500 MB | 437.7 | 856.6 | 2,974 | 4,369 | 22,679 | 46,679 | 1,418 MiB (0.996x file) |
| suisui Q4_0_XXS | 441.7 | 898.5 | 3,012 | 4,891 | 24,198 | 53,788 | 948 MiB (0.994x file) |
| suisui Q4_0_XS | 463.5 | 899.8 | 3,044 | 4,788 | 20,388 | 55,408 | 1,186 MiB (0.995x file) |
| suisui Q8_0_S | 459.8 | 942.1 | 2,946 | 4,161 | 24,999 | 56,181 | 1,425 MiB (0.996x file) |
| AtomicChat AD-IQ3_S | 460.4 | 907.8 | 2,622 | 4,481 | 19,636 | 56,525 | 873 MiB (0.993x file) |
| LiquidAI QAD-Q4_0 | 454.4 | 804.6 | 2,804 | 4,675 | 20,171 | 52,507 | 995 MiB (0.994x file) |
| AtomicChat AD-Q5_K_M | 485.1 | 906.0 | 3,010 | 4,409 | 24,265 | 47,061 | 1,244 MiB (0.995x file) |
| AtomicChat AD-Q6_K | 474.3 | 917.8 | 2,938 | 4,022 | 24,418 | 49,160 | 1,509 MiB (0.996x file) |
| LiquidAI Q8_0 | 452.5 | 920.1 | 2,966 | 4,188 | 26,380 | 53,215 | 1,755 MiB (0.997x file) |

## RTX Pro 6000 (Blackwell Server Edition)

| Checkpoint | Speculation | Batch 1 | Batch 2 | Batch 8 | Batch 16 | Prefill 2048 |
|---|---|---:|---:|---:|---:|---:|
| UD-Q4_K_XL | none | 67.8 | 122.6 | 344.0 | 502.3 | 4,736 |
| UD-Q4_K_XL | MTP, 2 draft tokens | 119.2 | 218.5 | 505.9 | 620.6 |  |
| UD-Q4_K_XL | MTP, 3 draft tokens | 121.6 | 223.9 | 544.4 | 583.6 |  |
| UD-Q4_K_XL | MTP, 4 draft tokens | 140.3 | 214.7 | 469.7 | 569.3 |  |
| UD-Q2_K_XL | none | 94.8 | 162.6 | 355.5 | 508.1 | 4,466 |
| UD-Q2_K_XL | MTP, 2 draft tokens | 150.5 | 254.2 | 508.8 | 631.0 |  |
| UD-Q2_K_XL | MTP, 3 draft tokens | 144.3 | 245.8 | 593.8 | 581.4 |  |
| UD-Q2_K_XL | MTP, 4 draft tokens | 153.6 | 229.9 | 512.2 | 554.2 |  |
