tcga_packages = {
    "pandas": "Data analysis and manipulation library (tables, dataframes).",
    "numpy": "Numerical computing with arrays, vectors, and matrices.",
    "matplotlib": "2D plotting and visualization library.",
    "scikit-learn": "Machine learning library for classification, regression, clustering, etc.",
    "pybedtools": "Genomic interval manipulation and analysis.",
    "lifelines": "Survival analysis library.",
    "scanpy": "Single-cell analysis library.",
    "scikit-bio": "Bioinformatics library.",
    "scipy": "Scientific computing library.",
    "scvelo": "Single-cell RNA velocity analysis.",
    "seaborn": "Statistical data visualization library.",
    "statsmodels": "Statistical modeling library.",
}

PYTHON_PACKAGE_SETS = {
    "tcga": tcga_packages,
    "kirc_closed_hypotheses": tcga_packages,  # closed task (experiment 06): same data as tcga
}
