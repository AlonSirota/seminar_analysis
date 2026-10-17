set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

source := "reproduce.py"
notebook := "reproduce.ipynb"
html_output := "reproduce.html"
pdf_output := "reproduce.pdf"

# Build and execute the notebook, embedding all cell outputs.
default: notebook

notebook:
    uv run jupytext --to ipynb "{{source}}" --output "{{notebook}}"
    uv run jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=600 "{{notebook}}"

# Export the executed notebook as a standalone HTML file.
html: notebook
    uv run jupyter nbconvert --to html "{{notebook}}" --output "{{html_output}}"

# Build the HTML export and open it in the default browser (macOS).
view: html
    open "{{html_output}}"

# Export the executed notebook as PDF without requiring LaTeX.
pdf: notebook
    uv run jupyter nbconvert --to webpdf --allow-chromium-download \
        "{{notebook}}" --output "{{pdf_output}}"
