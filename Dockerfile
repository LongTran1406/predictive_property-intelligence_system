# 1. Base image (Debian slim with Python 3.10)
FROM python:3.10.18-slim

# 2. Set environment variables
ENV PATH=/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LANG=C.UTF-8
ENV GPG_KEY=A035C8C19219BA821ECEA86B64E628F8D684696D
ENV PYTHON_VERSION=3.10.18
ENV PYTHON_SHA256=ae665bc678abd9ab6a6e1573d2481625a53719bc517e9a634ed2b9fefae3817f

# 3. Set work directory
WORKDIR /app

# 4. Copy files into container
COPY requirements.txt requirements.txt
COPY app.py app.py
COPY common common
COPY app-etl/tasks app-etl/tasks
COPY app-ml/src app-ml/src
COPY config/config.yaml config/config.yaml
COPY models/prod/latest_model models/prod/latest_model
COPY data/ml-ready/database_ml.parquet data/ml-ready/database_ml.parquet
COPY templates/ templates/
COPY data/test/Test.parquet data/test/Test.parquet
COPY static/ static/
COPY data/avg/avg_info.json data/avg/avg_info.json

# 5. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Default command
CMD ["python", "app.py"]