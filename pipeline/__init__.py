# pipeline/__init__.py
# Makes pipeline/ a Python package.
# Individual modules are imported by run_all.py using lazy imports
# to avoid triggering heavy model loads on --help or --list-steps.
