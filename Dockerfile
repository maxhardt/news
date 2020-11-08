FROM tiangolo/uvicorn-gunicorn:python3.8

# set up environment
RUN mkdir ing
WORKDIR /ing

# setup python environment
COPY ./requirements.txt .
RUN pip install "--upgrade" pip --trusted-host pypi.python.org
RUN pip install "--no-cache-dir" -r "requirements.txt" --trusted-host pypi.python.org

# setup app
COPY ./.env .
COPY ./pipeline.yaml .
COPY ./app.py .
COPY ./run_ml.py .
RUN mkdir src
COPY ./src src

# entry point
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
