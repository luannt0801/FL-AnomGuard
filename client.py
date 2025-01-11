from paho.mqtt.client import Client as MqttClient
from paho.mqtt.client import MQTTv311
from src.utils import ping_host
from src.logging_setting import logger
from src.lstm import start_training_task

import json

class Client(MqttClient):
    def __init__(self, client_id="", host="", port =int, clean_session=None, userdata=None, protocol=MQTTv311, transport="tcp", reconnect_on_failure=True):
        super().__init__(client_id, clean_session, userdata, protocol, transport, reconnect_on_failure)
        self.client_id = client_id
        self.host = host
        self.port = port
        print(f"Starting on client_{client_id}")
        
    def on_connect(self, client, userdata, flags, rc):
        logger.debug(f"do on_connect")
        print(f"Connected with result code {rc}")
        self.publish(topic="FL/join", payload=self.client_id)
        print(f"client_{self.client_id} join FL/join of {self.host}")

    def on_disconnect(self, client, userdata, flags, rc):
        logger.debug(f"do on_connect")
        self.reconnect()

    def on_message(self, client, userdata, msg):
        logger.debug(f"do on_message")
        print(f"on_message {self._client_id.decode()}")
        print(f"RECEIVED msg from {msg.topic}")
        topic = msg.topic
        print(topic)
        if topic == "FL/req/" + self.client_id:
            self.handle_task(msg)

    def on_subcribe(self, mid, granted_qos):
        logger.debug(f"do on_subcribe")
        print(f"Subcribe: {mid}, {granted_qos}")

    def do_evaluate_connection(self):
        logger.debug(f"do evaluate_connection")
        result = ping_host(self.host)
        result["client_id"] = self.client_id
        result["task"] = "EVALUATION_CONNECT_DONE"
        self.publish(
            topic="FL/res/" + self.client_id, payload=json.dumps(result)
        )
        print(f"Published to topic FL/res/{self.client_id}")
        return result

    def do_train(self):
        logger.debug(f"do do_train")
        result = start_training_task()
        result_np = {key: value.cpu().numpy().tolist() for key, value in result.items()}
        payload = {
            "task": "DONE_TRAIN",
            "weight": result_np
        }

        self.publish(topic="FL/res/" + self.client_id, payload=json.dumps(payload))
        print("End trainning")

    def do_stop_client(self):
        logger.debug(f"do on_connect")
        self.loop_stop()

    def handle_task(self, msg):
        logger.debug(f"handle_task")
        task_name = msg.payload.decode("utf-8")
        if task_name == "EVALUATION_CONNECT":
            self.do_evaluate_connection()
        elif task_name == "START_TRAIN":
            self.do_train()
        elif task_name == "STOP":
            self.do_stop_client()
        else:
            print(f"Command {task_name} is not supported")

    def handle_model(self, client, userdata, msg):
        """
        Process the model received from the server and save it to the file client_model.pt
        Respond to the server with the message: WRITE_MODEL
        This function is always executed when the client starts.
        """
        logger.debug(f"handle_model")
        print(f"Receive model from server")

        with open("src/parameters/client.pt", "wb") as f:
            f.write(msg.payload)
        msg_send_server = {
            "client_id": self.client_id,
            "task": "CLIENT_RECEIVE_MODEL"
        }
        self.publish(topic="FL/res/" + self.client_id, payload=json.dumps(msg_send_server))

    def start(self):
        logger.debug(f"client start")
        self.connect(host=self.host, port=self.port, keepalive=3600)
        self.message_callback_add("FL/model/all_client", self.handle_model)
        self.loop_start()

        self.subscribe(topic="FL/model/all_client")
        self.subscribe(topic="FL/req/" + self.client_id)
        self.publish(topic="FL/join", payload=self.client_id)

        print(f"{self.client_id} joined FL/join of {self.host}")

        self._thread.join()
        print("Client exits")

client = Client(client_id="1", host='192.168.100.105', port=1883)
client.start()


        
