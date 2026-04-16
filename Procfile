web: gunicorn app:app --bind 0.0.0.0:${PORT:-8080} --workers 1 --worker-class sync --threads 1 --timeout 180
