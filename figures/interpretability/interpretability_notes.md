# Interpretability Notes (Paper-Ready)

- Script analyzed: Devanagari
- Sample image: `DEVNAGARI_NEW/TEST/1/001_01.jpg`
- Model pipeline: Input image -> CNN feature map -> KAN nonlinear transfer -> class prediction

## What each KAN curve represents
Each plotted curve is one KAN hidden unit transfer function. For unit `u`, the x-axis is the affine input `z_u = w_u^T f + b_u` from CNN features, and the y-axis is `z_u + sin(alpha_u * z_u)`.
If `alpha_u` is larger, the curve oscillates more rapidly, indicating stronger nonlinearity. The green marker in each subplot is the current sample's operating point.

## Top predictions
1. Class 1: 99.31%
2. Class 22: 0.12%
3. Class 9: 0.12%

## Most influential KAN units in this sample
Units were selected by largest absolute transformed activation.
- Unit 82: |activation|=2.9965, alpha=1.0368
- Unit 87: |activation|=2.9262, alpha=0.9912
- Unit 62: |activation|=2.9048, alpha=1.0847
- Unit 26: |activation|=2.8500, alpha=1.1168
- Unit 99: |activation|=2.8302, alpha=1.1058
- Unit 84: |activation|=2.8242, alpha=1.1149

## Suggested figure captions
- `pipeline_summary.png`: End-to-end interpretable flow from input to decision through CNN and KAN.
- `cnn_feature_maps.png`: Representative CNN channel activations highlighting spatial evidence used for recognition.
- `kan_transfer_curves.png`: Unit-wise KAN nonlinear transfer functions with sample operating points.
- `kan_all_splines_single_diagram.png`: Overlay of all KAN unit transfer curves in one plot (color indicates alpha).
- `feature_transformation.png`: Relationship between affine unit input and KAN output, showing non-linear shaping.
