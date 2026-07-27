.PHONY: install test train predict clean lint

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest tests/ -v --tb=short

train:
	python -m src.models.train_model

predict:
	python -m src.models.predict_model --input data/processed/housing_clean.csv

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".DS_Store" -delete

lint:
	flake8 src/ tests/ --max-line-length=100
