"""
conftest.py — root-level pytest configuration.
Placed in the project root so it applies to all test runs.
"""

collect_ignore_glob = [
    "*.txt",  # test_output.txt, test_run_output.txt, error_log.txt etc.
    "*.log",
    "*.bat",
    "*.json",
    "barometer_saved/**",
    "sentiment_cache/**",
    ".venv/**",
    "dashboard_static/**",
]
