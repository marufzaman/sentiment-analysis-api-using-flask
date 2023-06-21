FROM python:3.11-buster AS builder

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir setuptools mercurial --upgrade \
    && pip install --no-cache-dir -r /tmp/requirements.txt

FROM python:3.11-slim-buster AS runner

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

RUN apt update \
    && apt upgrade -y \
    && apt install -y libpq5 libxml2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && apt autoremove -y

RUN pip install --no-cache-dir gunicorn==20.1.0