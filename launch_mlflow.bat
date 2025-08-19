call cd /d %~dp0 
call uv run mlflow server --backend-store-uri runs/mlflow --host 0.0.0.0 --port 5000