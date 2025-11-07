# prometheus_exporter.py
from prometheus_client import start_http_server, Gauge, Counter, Summary
import random
import time

# Metrik utama: output prediksi model
prediction_metric = Gauge('model_predictions', 'Example prediction output from model')

# Tambahan metrik:
inference_counter = Counter('mlflow_inference_requests_total', 'Total number of inference requests')
inference_duration = Summary('mlflow_inference_duration_seconds', 'Time taken for a single inference')
accuracy_metric = Gauge('model_accuracy_score', 'Dummy accuracy score of the model')

# Jalankan server Prometheus di port 8000
start_http_server(8000)

@inference_duration.time()
def simulate_inference():
    # Dummy prediksi (0 atau 1)
    pred = random.choice([0, 1])
    prediction_metric.set(pred)
    inference_counter.inc()

    # Dummy accuracy (acak antara 0.85 - 0.95)
    accuracy = round(random.uniform(0.85, 0.95), 4)
    accuracy_metric.set(accuracy)

if __name__ == '__main__':
    while True:
        simulate_inference()
        time.sleep(5)
