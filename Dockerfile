FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

WORKDIR /app

ENTRYPOINT ["python", "run_survey_and_hybrid_pipeline.py"]
CMD ["--fe-config", "feature_engineering/feature_engineering_config.json", "--train-config", "survey_and_hybrid_approach_pipeline/config.json"]
