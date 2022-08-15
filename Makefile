install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	#python -m pytest -vv test.py

lint:
	#pylint --disable=R,C,global-variable-undefined,no-member main.py

all: install lint test 
