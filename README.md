# 2-Phase Simplex LP Solver

## Installation
- Requirements:
    - python>=3.9.0
- Create a virtualenv and activate it.
```shell
pip install virtualenv
virtualenv venv

# On Windows:
venv/Scripts/activate
# On Linux:
source venv/bin/activate
```

- Install the requirements using pip command:

```shell
pip install -r requirements.txt
```

- To run the code, use the following command:

```shell
python main.py <filename> # Normal output
python main.py <filename> -v # For verbosity

# Example:
python main.py test1.lp -v
```