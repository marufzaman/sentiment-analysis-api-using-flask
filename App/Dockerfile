# Builder stage
FROM python:3.11-alpine AS builder

WORKDIR /app

RUN apk update \
    && apk add --no-cache --virtual .build-deps gcc g++ musl-dev libffi-dev openssl-dev python3-dev cargo

COPY App/requirements.txt .

RUN python3 -m venv venv \
    && /app/venv/bin/python -m pip install --no-cache-dir --upgrade pip \
    && /app/venv/bin/python -m pip install --no-cache-dir numpy==1.23.2 \
    && /app/venv/bin/python -m pip install --no-cache-dir -r requirements.txt

# Runner stage
FROM python:3.11-alpine AS runner

WORKDIR /app

RUN apk update \
    && apk add --no-cache libpq libxml2 \
    && python3 -m venv venv

COPY --from=builder /app/venv /app/venv
COPY --from=builder /app/app.py /app/app.py
COPY --from=builder /app/custom-model /app/custom-model
COPY --from=builder /app/static /app/static
COPY --from=builder /app/templates /app/templates

RUN apk add --no-cache glibc-dev libstdc++

RUN pip install --no-cache-dir gunicorn==20.1.0

# Set the entry point to activate the virtual environment and run the program
ENTRYPOINT ["/app/venv/bin/python", "/app/app.py"]

EXPOSE 5000
