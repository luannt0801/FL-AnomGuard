import ping3
import numpy as np
import yaml

def get_config():
    path = "config.yaml"
    with open(path, "r") as stream:
            config = yaml.load(stream, Loader=yaml.Loader)
    server_config = config['server']
    client_config = config['client']
    data_config = config['client']['data']
    return server_config, client_config, data_config

server_config, client_config, data_config = get_config()

def ping_host(host, count=10):
    ping_result = [ping3.ping(host) for _ in range(count)]
    ping_result = [
        result for result in ping_result if result is not None
    ]  # Loại bỏ các kết quả None (không thành công)

    if ping_result:
        avg_latency = sum(ping_result) / len(ping_result)
        min_latency = min(ping_result)
        max_latency = max(ping_result)
        packet_loss = (1 - len(ping_result) / count) * 100
    else:
        avg_latency = None
        min_latency = None
        max_latency = None
        packet_loss = 100

    return {
        "host": host,
        "avg_latency": avg_latency,
        "min_latency": min_latency,
        "max_latency": max_latency,
        "packet_loss": packet_loss,
    }

