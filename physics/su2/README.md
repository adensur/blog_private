# Physics Journal

This folder is a Quarto website for working notes on physics and mathematics.

## Local workflow

Install the Python packages used by executable Quarto chunks:

```bash
python3 -m pip install --user --break-system-packages -r requirements.txt
```

Render the static site:

```bash
quarto render
```

Open the rendered journal:

```bash
open ../../docs/index.html
```

For a live preview while writing:

```bash
quarto preview
```

The matrix playground is embedded with Shinylive, so its app code is Python but runs in the browser. Use `quarto preview` when editing so the local web server can serve the generated Shinylive assets.

If a fresh terminal cannot find `quarto`, reload the shell with `source ~/.zshrc`.

## GitHub Pages

Live site: https://adensur.github.io/blog_private/

This project renders to the repository-root `docs/` directory. In GitHub, Pages is configured to publish from the `main` branch and `/docs` folder.
