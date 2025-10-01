# -*- coding: utf-8 -*-
import os
from datetime import datetime, timezone
try:
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
except Exception:
    CollectorRegistry = Gauge = push_to_gateway = None

def push_business_metrics(
    site: str, orientation: str,
    rmse: float, mape: float,
    y_last: float, yhat_last: float,
    last_ts: datetime,
    gateway: str = None,
) -> None:
    if push_to_gateway is None:
        return
    if isinstance(last_ts, datetime) and last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=timezone.utc)

    reg = CollectorRegistry()
    Gauge('bike_rmse', 'RMSE', ['site','orientation'], registry=reg).labels(site, orientation).set(rmse)
    Gauge('bike_mape', 'MAPE_%', ['site','orientation'], registry=reg).labels(site, orientation).set(mape)
    Gauge('bike_y_last', 'Observed last', ['site','orientation'], registry=reg).labels(site, orientation).set(y_last)
    Gauge('bike_yhat_last', 'Predicted last', ['site','orientation'], registry=reg).labels(site, orientation).set(yhat_last)
    Gauge('bike_last_ts_epoch', 'Last timestamp epoch', ['site','orientation'], registry=reg)\
        .labels(site, orientation).set(int(last_ts.timestamp()))

    gw = gateway or os.getenv('PUSHGATEWAY_ADDR', 'monitoring-pushgateway:9091')
    push_to_gateway(gw, job='bike-traffic', registry=reg)
