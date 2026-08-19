# Genetic perturbation scRNA-seq application

Application-specific preprocessing, source/target definitions, evaluation, and
validation scripts will be added from the corresponding source analysis. The
shared HurdleAE/CycleGAN implementation must be imported from `cell_translator`
and must not be redefined here.

The final runner should use both `--gene` and `--config`. Each gene-specific YAML
file should contain placeholder data paths, source/target states, and only the
logic/settings that genuinely differ by perturbation.
