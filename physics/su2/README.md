# Physics Journal

This folder is a Quarto website for working notes on physics and mathematics.

## Local workflow

Render the static site:

```bash
~/.local/bin/quarto render
```

Open the rendered journal:

```bash
open ../../docs/index.html
```

For a live preview while writing:

```bash
~/.local/bin/quarto preview
```

Regenerate the transformation figure:

```bash
python3 scripts/generate_transformation_figures.py
```

On this machine, Quarto was also installed without `sudo` at:

```bash
~/.local/bin/quarto
```

## GitHub Pages

This project renders to the repository-root `docs/` directory. In GitHub, configure Pages to publish from the `main` branch and `/docs` folder.
