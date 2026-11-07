"""All thresholds and timeouts in one place."""

PHASE1_WORKERS = 8
CODING_TIME_LIMIT = 90
PHASE1_TIMEOUT = {"coding": 100, "default": 15}
LLM_CONCURRENCY = 3  # provider account ceiling: 3 concurrent requests (HTTP 429 above it)
LLM_TIMEOUT = 180
DEADLINE_S = 3300
STOP_NEW_BEFORE_DEADLINE_S = 150
RESERVE_FRACTION = 0.01
TIMEOUT_CHARGE = 1500  # charged for an LLM call that ended by timeout
