pip uninstall fewshot-prp -y

rm -rf build/
rm -rf dist/
rm -rf fewshot_prp.egg-info/

find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null

pip install -e .