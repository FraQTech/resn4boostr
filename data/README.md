# Data placement

This repository does not include BOOSTR data files. The CSV of the BOOSTR Partial Release dataset used in this project is available at https://zenodo.org/records/4088982

Place the partial release CSV in the repository root:

```text
BOOSTR_PartialRelease.csv
```

or pass a full path with `--data`.

Do not commit large raw datasets, full BOOSTR archives, or generated run folders to GitHub. They are ignored by `.gitignore`.
