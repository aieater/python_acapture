find * -name "*.pyc" | xargs rm -f
find * -name "*.egg-info" | xargs rm -rf
rm -rf build
rm -rf dist
find * -name "__pycache__" | xargs  rm -rf
