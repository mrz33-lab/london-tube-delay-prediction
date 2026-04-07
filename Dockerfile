FROM python:3.11-slim

WORKDIR /app

# I copy the dependency list first so Docker can cache this layer independently
# of source code changes — reinstalling packages only when requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# I copy the rest of the project files after the dependency layer
COPY . .

# I create a non-root user so the process runs with minimal privileges,
# which reduces the blast radius of any container escape vulnerability.
RUN useradd --no-create-home --shell /bin/false appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Use the stdlib urllib so curl is not needed in the slim image.
# start-period gives the model time to load before the first health check.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request, sys; \
r = urllib.request.urlopen('http://localhost:8000/health', timeout=8); \
sys.exit(0 if r.status == 200 else 1)"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
