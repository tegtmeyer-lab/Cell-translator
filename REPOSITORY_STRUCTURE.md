# Recommended GitHub organization

```text
Cell-Translator/
├── README.md
├── LICENSE                         # add the license chosen for public release
├── pyproject.toml
├── requirements.txt               # pin exact manuscript versions before release
├── .gitignore
│
├── cell_translator/                # ONE canonical implementation
│   ├── __init__.py
│   ├── hyperparameters.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── hurdle_ae.py
│   │   └── cyclegan.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── hurdle_ae_training.py
│   │   └── cyclegan_training.py
│   ├── inference/
│   │   ├── __init__.py
│   │   └── translate.py
│   └── utils/
│       ├── __init__.py
│       ├── reproducibility.py
│       └── validation.py
│
├── applications/
│   ├── neuronal_differentiation/
│   │   ├── run.py
│   │   └── config.yaml
│   ├── cerebral_organoid/
│   │   ├── prepare_data.py         # add from exact source analysis
│   │   ├── run.py
│   │   ├── evaluate.py
│   │   ├── validate.py
│   │   └── config.yaml
│   ├── pancreatic_endocrine/
│   │   └── ...
│   ├── hematopoietic_differentiation/
│   │   └── ...
│   ├── leukemia_transformation/
│   │   └── ...
│   └── genetic_perturbation/
│       ├── prepare_data.py
│       ├── run.py                  # argparse: --gene + --config
│       ├── evaluate.py
│       └── validate.py
│
├── configs/
│   └── genetic_perturbation/
│       ├── apoe.yaml
│       ├── nr1h3.yaml
│       ├── msi2.yaml
│       ├── lrrc4c.yaml
│       └── ankrd12.yaml
│
├── data/
│   └── README.md                   # accessions + expected processed-file schema
│
├── scripts/
│   └── reproduce_all.sh            # add after all 6 application runners are finalized
│
└── tests/
    ├── test_hurdle_ae.py
    └── test_cyclegan_shapes.py
```

The current ZIP intentionally contains only the finalized shared core plus the
neuronal application runner. Empty/README-only application folders are included
as organization placeholders; their scientific logic should not be invented.
