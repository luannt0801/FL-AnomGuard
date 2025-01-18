from paho.mqtt.client import Client as MqttClient
from paho.mqtt.client import MQTTv311
from src.utils import ping_host
from src.logging_setting import logger
from src.lstm import start_training_task
from src.cnn import start_trainning_CNN
from src.data import Data, get_dataloader
from src.utils import (server_config, client_config, data_config)
from collections import Counter
import json
import torch

class Client(MqttClient):
    def __init__(self, client_id="", host="", port =int, clean_session=None, userdata=None, protocol=MQTTv311, transport="tcp", reconnect_on_failure=True):
        super().__init__(client_id, clean_session, userdata, protocol, transport, reconnect_on_failure)
        self.client_id = client_id
        self.host = host
        self.port = port
        self.trainloader = None
        self.testloader = None
        logger.debug(f"client_{client_id}")
    
    def on_connect(self, client, userdata, flags, rc):
        logger.debug(f"Connected with result code {rc}")
        self.publish(topic="FL/join", payload=self.client_id)
        logger.warning(f"client_{self.client_id} join FL/join of {self.host}")

    def on_disconnect(self, client, userdata, flags, rc):
        logger.debug(f"on_disconnect with code {rc}")
        if rc != 0:  # Chỉ tự động kết nối lại nếu mất kết nối bất thường
            logger.warning("Unexpected disconnection. Trying to reconnect...")
            self.reconnect()
        else:
            logger.info("Disconnected gracefully.")

    def on_message(self, client, userdata, msg):
        logger.debug(f"on_message")
        # logger.debug(f"on_message {self._client_id.decode()}")
        # logger.debug(f"RECEIVED msg from {msg.topic}")
        topic = msg.topic
        # logger.debug(topic)
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
        logger.warning(f"Published to topic FL/res/{self.client_id}")
        return result

    def do_train(self):
        logger.debug(f"\ndo_train")
        if data_config['name_data'] == 'dga':
            try:
                result = start_training_task(trainloader=self.trainloader, testloader=self.testloader)
            except Exception as e:
                logger.error(f"Error during training: {str(e)}")
        elif data_config['name_data'] == 'cifar10':
            try:
                result = start_trainning_CNN(trainloader=self.trainloader, testloader=self.testloader)
            except Exception as e:
                logger.error(f"Error during training: {str(e)}")

        if result is not None:
            result_np = {key: value.cpu().numpy().tolist() for key, value in result.items()}
        else:
            logger.error("Result is None!")

        payload = {
            "task": "DONE_TRAIN",
            "weight": result_np
        }

        self.publish(topic="FL/res/" + self.client_id, payload=json.dumps(payload))
        logger.warning("End trainning")

    def do_stop_client(self):
        logger.debug(f"do_stop_client")
        self.loop_stop()

    def handle_task(self, msg):
        logger.debug(f"do handle_task")
        task_name = msg.payload.decode("utf-8")
        logger.warning(f"Received task name: {task_name}")
        if task_name == "EVALUATION_CONNECT":
            self.do_evaluate_connection()
        elif task_name == "SELF_CLUSTER":
            self.do_selflearning_data()
        elif task_name == "UPDATE_LABEL":
            self.do_update_generalLabel()
        elif task_name == "START_TRAIN":
            self.do_train()
        elif task_name == "STOP":
            self.do_stop_client()
        else:
            logger.warning(f"Command {task_name} is not supported")

    def do_selflearning_data(self):
        self.trainloader, self.testloader = get_dataloader()
        logger.debug(f"do_selflaerning_data")
        payload = {
            'task': 'DONE_SELF_CLUSTER'
        }
        payload_str = json.dumps(payload)
        self.publish(topic='FL/res/'+str(self.client_id), payload=payload_str)

    def do_update_generalLabel(self):
        logger.debug(f"do_update_generalLabel")
        payload = {
            'task': 'DONE_UPDATE_LABEL'
        }
        payload_str = json.dumps(payload)
        self.publish(topic='FL/res/'+str(self.client_id), payload=payload_str)

    def handle_model(self, client, userdata, msg):
        """
        Process the model received from the server and save it to the file client_model.pt
        Respond to the server with the message: WRITE_MODEL
        This function is always executed when the client starts.
        """
        logger.debug(f"handle_model")
        logger.warning(f"Receive model from server")

        with open("src/parameters/client.pt", "wb") as f:
            f.write(msg.payload)
        msg_send_server = {
            "client_id": self.client_id,
            "task": "CLIENT_RECEIVE_MODEL"
        }
        self.publish(topic="FL/res/" + self.client_id, payload=json.dumps(msg_send_server))

    def start(self):
        logger.debug(f"client start")
        self.connect(host=self.host, port=self.port, keepalive=65535)
        self.message_callback_add("FL/model/all_client", self.handle_model)
        self.loop_start()

        self.subscribe(topic="FL/model/all_client")
        self.subscribe(topic="FL/req/" + self.client_id)
        self.publish(topic="FL/join", payload=self.client_id)

        logger.debug(f"{self.client_id} joined FL/join of {self.host}")

        self._thread.join()
        logger.debug("Client exits")

client = Client(client_id=client_config['client_id'], host=client_config['host'], port=1883)
client.start()