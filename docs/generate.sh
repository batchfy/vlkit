rm -rf build
rm source/vlkit*.rst
sphinx-apidoc -o source ../vlkit/
make html
