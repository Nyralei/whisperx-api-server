"""Observability package: Prometheus metrics shims, registry, and taxonomy.

When METRICS_ENABLED is unset or false, only the null-object shim classes
in pipeline.py, http.py, gpu.py, and kafka.py are imported. prometheus_client
is never imported unless setup_metrics() is called from create_app().
"""
