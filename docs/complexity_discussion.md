# Complexity Analysis Write-up

This note provides manuscript-ready wording and an experimental protocol for adding a computational complexity discussion to the KeyRe-ID paper. Replace the placeholders in square brackets after running the benchmark.

## Manuscript Paragraph

To further assess the computational cost of the proposed method, we report the model complexity and inference efficiency under a unified evaluation protocol. All measurements are conducted at the clip level using an input clip of four RGB frames with a spatial resolution of 256 x 128, together with the corresponding six-channel keypoint heatmaps. Unless otherwise stated, the batch size is set to 1 and the model is evaluated in inference mode without gradient computation. We measure the number of parameters, floating-point operations (FLOPs), and wall-clock inference latency. FLOPs are computed for a single forward pass of one 4-frame clip, while inference latency is averaged over [N] repeated runs after [W] warm-up iterations to reduce the effect of GPU initialization and kernel compilation overhead. CUDA synchronization is applied before and after timing to ensure accurate measurement.

The additional computational cost of KeyRe-ID mainly comes from two components: the keypoint-guided part segmentation branch and the temporal clip shift-and-shuffle operation. The keypoint-guided branch introduces part-aware feature aggregation and six local identity streams, whereas TCSS reorganizes patch tokens along the temporal dimension to enhance robustness against temporal misalignment. Despite these additional operations, the overall overhead remains moderate because the heatmap-based weighting is performed on patch-level representations rather than on dense pixel-level feature maps, and TCSS itself does not introduce additional learnable parameters. As shown in Table [X], KeyRe-ID requires [X]M parameters and [X]G FLOPs per clip, with an average inference time of [X] ms on [GPU name]. Compared with the baseline video transformer, the proposed method increases the computational cost by [X]%, while improving Rank-1 accuracy and mAP by [X]% and [X]%, respectively. These results indicate that the proposed part-aware temporal modeling provides a favorable trade-off between recognition accuracy and inference efficiency.

## Experimental Setting

Use the following setting when reporting complexity:

| Item | Setting |
|---|---|
| Input image size | 256 x 128 |
| Clip length | 4 frames |
| RGB input shape | 1 x 4 x 3 x 256 x 128 |
| Heatmap input shape | 1 x 4 x 6 x 256 x 128 |
| Batch size | 1 |
| Mode | `model.eval()` |
| Gradient | Disabled with `torch.no_grad()` |
| Warm-up iterations | 50 recommended |
| Timed iterations | 200 recommended |
| Timing method | `torch.cuda.Event` with `torch.cuda.synchronize()` |
| Reported latency | Mean +/- standard deviation in ms/clip |
| Throughput | Clips per second or frames per second |
| Hardware | GPU name, CUDA version, PyTorch version |

Recommended sentence for reproducibility:

> All latency results are measured on a single [GPU name] using PyTorch [version] and CUDA [version]. We report the average inference time over 200 runs after 50 warm-up iterations with batch size 1.

## Table Template

Use this table in the paper after running the benchmark:

| Method | Params (M) | FLOPs (G/clip) | Latency (ms/clip) | Throughput (clips/s) | Rank-1 (%) | mAP (%) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline video transformer | [X] | [X] | [X] | [X] | [X] | [X] |
| Baseline + temporal attention | [X] | [X] | [X] | [X] | [X] | [X] |
| Baseline + KPS | [X] | [X] | [X] | [X] | [X] | [X] |
| Baseline + KPS + TCSS (KeyRe-ID) | [X] | [X] | [X] | [X] | [X] | [X] |

If ablation models are not available, use the simpler table below:

| Method | Params (M) | FLOPs (G/clip) | Latency (ms/clip) | Throughput (clips/s) | Rank-1 (%) | mAP (%) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline video transformer | [X] | [X] | [X] | [X] | [X] | [X] |
| KeyRe-ID | [X] | [X] | [X] | [X] | [X] | [X] |

## Reviewer Response

We thank the reviewer for the valuable suggestion. We have added a computational complexity analysis to the revised manuscript. Specifically, we report the number of parameters, FLOPs per 4-frame clip, and average inference latency measured under a unified setting with batch size 1. We also discuss the additional cost introduced by the keypoint-guided part segmentation branch and TCSS. The new results show that KeyRe-ID introduces only a moderate computational overhead compared with the baseline video transformer, while providing consistent improvements in Rank-1 accuracy and mAP. This demonstrates that the proposed method achieves a favorable balance between recognition performance and inference efficiency.

## Notes For Filling Values

When filling the placeholders, avoid comparing latency numbers measured under different hardware or software environments. FLOPs and parameter counts can be compared across machines if the same input resolution, clip length, and counting tool are used. Latency should always be reported together with the GPU model, CUDA version, PyTorch version, batch size, warm-up count, and number of timed iterations.

If the FLOP counting tool reports unsupported operators, mention this explicitly:

> FLOPs are estimated using [tool name]. Operators unsupported by the tool are excluded from the reported FLOP count; therefore, the value should be interpreted as an approximate measure of computational complexity.

If only inference latency is available, use the following conservative wording:

> Since practical deployment cost is also affected by memory access and GPU kernel efficiency, we additionally report wall-clock inference latency, which directly reflects the end-to-end cost of processing one input clip.
