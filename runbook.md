# Runbook — HYDRA Prime (executive summary)
1. Read README.
2. Build Docker image: docker build -t hydra-agent:latest .
3. Start locally: docker-compose up --build
4. Replace placeholder images with your registry and push.
5. Deploy to Kubernetes (example): kubectl apply -f k8s/deployment.yaml
6. Configure HPA with metrics server and set maxReplicas to 1000 when ready.
7. Set up Prometheus to scrape /metrics endpoints from agents; set alerting rules for failure rates.
8. For cloning logic: run the 'scaler' script (not included) which queries monitoring and triggers new deployments with mutated env vars.
9. Security: store API keys in secret manager; never commit them.
10. Cost control: start small; put budget alarms on cloud accounts.
