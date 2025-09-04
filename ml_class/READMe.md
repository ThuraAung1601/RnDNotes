
### MNIST Experiment Result

| Model                 | Top1 Accuracy (%) | Accuracy (%) | Inference Time (s) |
|-----------------------|-----------------|-----------------|------------------|
| Teacher               | 99.18           | 99.18           | 0.001335         |
| Student               | 98.48           | 98.48           | 0.000816         |
| Vanilla KD (T=1.0)    | 98.50           | 98.50           | 0.000816         |
| Feature KD (λ=2.0)    | 98.51           | 98.51           | 0.000918         |
| Sim KD                | 98.19           | 98.19           | 0.001032         |
