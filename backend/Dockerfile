FROM python:3.9-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    proj-bin \
    proj-data \
    libspatialindex-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt build-constraints.txt ./

ENV PIP_CONSTRAINT=/app/build-constraints.txt

RUN python -m venv /opt/scout-env \
    && /opt/scout-env/bin/pip install --upgrade pip \
    && /opt/scout-env/bin/pip install "setuptools<70" "wheel" "cython<3" "jinja2<3.2" \
    && /opt/scout-env/bin/pip install "numpy==1.23.5" \
    && /opt/scout-env/bin/pip install --no-build-isolation "rasterio==1.3.11" \
    && /opt/scout-env/bin/pip install -r requirements.txt

COPY . .

RUN ln -s /opt/scout-env /app/envs

EXPOSE 5000

CMD ["/bin/sh", "-lc", "ln -sfn /opt/scout-env /app/envs && /opt/scout-env/bin/python server.py"]
