\# W12D2 — Containerising ML Apps



\## Overview



This project demonstrates how to containerise a machine learning API using Docker and automate its CI/CD workflow using GitHub Actions.



\## Technologies Used



\* Python 3.12

\* FastAPI

\* Docker

\* GitHub Actions

\* Ruff

\* Pytest

\* Docker Hub



\## Project Structure



```text

W12D2\_Docker\_ML\_App/

├── app/

├── tests/

├── .github/

│   └── workflows/

│       └── ci.yml

├── .dockerignore

├── Dockerfile

├── MONITORING.md

├── README.md

├── requirements.txt

└── requirements-dev.txt

```



\## Docker



Build the Docker image:



```bash

docker build -t w12d2-ml-api:latest .

```



Run the container:



```bash

docker run -p 8000:8000 --name w12d2-ml-api-container w12d2-ml-api:latest

```



The API can then be accessed at:



```text

http://localhost:8000

```



FastAPI documentation:



```text

http://localhost:8000/docs

```



\## Testing



Run the test suite with:



```bash

python -m pytest -q

```



Run linting with:



```bash

ruff check .

```



\## CI/CD



GitHub Actions automates the following stages:



```text

Lint → Test → Build Docker Image → Push Docker Image

```



The Docker image is pushed to Docker Hub only when changes are pushed to the `main` branch.



Docker Hub credentials are stored securely as GitHub repository secrets:



\* `DOCKERHUB\_USERNAME`

\* `DOCKERHUB\_TOKEN`



\## Monitoring



The monitoring strategy for the ML API covers application health, model behaviour, container health, CI/CD pipeline status, logs, latency, errors, and potential data drift.



See `MONITORING.md` for details.



