import paho.mqtt.client as mqtt
import json
import torch
import threading
import time

from paho.mqtt.client import Client as MqttClient
from collections import OrderedDict
from src.logging_setting import logger

class Server(MqttClient):
    def __init__(self, client_fl_id, clean_session=True, userdata=None, protocol=mqtt.MQTTv311):
        super().__init__(client_fl_id, clean_session, userdata, protocol)

        self.client_dict = {}
        self.client_trainres_dict = {}

        self.numRound = 10 # input
        self.numDevice = 1 # input
        self.round_state = "finished"
        self.round = 0

    def on_connect(self, client, userdata, flags, rc): # define by paho-mqtt
        logger.debug(f"do on_connect")
        print("Connected with result code " + str(rc))
    
    def on_disconnect(self, client, userdata, flags, rc):
        logger.debug(f"do on_disconnect")
        print("Disconnected with result code " + str(rc))
        self.reconnect()

    def on_message(self, client, userdata, msg):
        logger.debug(f"do on_message")
        topic = msg.topic
        if topic == "FL/join": 
            this_client_id = msg.payload.decode("utf-8")
            print("joined from" + " " + this_client_id)
            self.client_dict[this_client_id] = {"state": "joined"}
            print("Check server join topic res")
            self.subscribe(topic="FL/res/" + this_client_id)
        elif "FL/res" in topic:
            tmp = topic.split("/")
            this_client_id = tmp[2]
            self.handle_res(this_client_id, msg)

    def on_subscribe(self, client, userdata, mid, granted_qos):
        logger.debug(f"do on_subcribe")
        print("Subscribed: " + str(mid) + " " + str(granted_qos))

    def send_task(self, task_name, this_client_id):
        print("publish to " + "FL/req/" + this_client_id)
        self.publish(topic="FL/req/" + this_client_id, payload=task_name)

    def send_model(self, path, this_client_id):
        logger.debug(f"do send_model")
        f = open(path, "rb")
        data = f.read()
        f.close()
        self.publish(topic="FL/model/all_client", payload=data)

    def handle_res(self, this_client_id, msg):
        data = json.loads(msg.payload)
        cmd = data["task"]
        logger.warning(f"Check the cmd: {cmd}")
        if cmd == "EVALUATION_CONNECT_DONE":
            print(f"{this_client_id} complete task EVA_CONN")
            self.handle_pingres(this_client_id, msg)
        elif cmd == "DONE_TRAIN":
            print(f"{this_client_id} complete task TRAIN")
            self.handle_trainres(this_client_id, msg)
        elif cmd == "CLIENT_RECEIVE_MODEL":
            print(f"{this_client_id} complete task WRITE_MODEL")
            self.handle_update_writemodel(this_client_id, msg)

    def handle_pingres(self, this_client_id, msg):
        logger.debug(f"do handle_pingres")
        ping_res = json.loads(msg.payload)
        this_client_id = ping_res["client_id"]

        if ping_res["packet_loss"] == 0.0:
            print(f"{this_client_id} is a good client")
            state = self.client_dict[this_client_id]["state"]
            print(f"state {this_client_id}: {state}, round: {self.round}")
            if state == "joined" or state == "trained":
                self.client_dict[this_client_id]["state"] = "eva_conn_done"
                count_eva_conn_ok = sum(1 for client_info in self.client_dict.values() if client_info["state"] == "eva_conn_ok")
                self.send_model("src/parameters/server.pt", this_client_id)

    def handle_trainres(self, this_client_id, msg):
        logger.debug(f"do handle_trainres")
        payload = json.loads(msg.payload.decode())

        self.client_trainres_dict[this_client_id] = payload["weight"]
        state = self.client_dict[this_client_id]["state"]
        if state == "model_recv":
            self.client_dict[this_client_id]["state"] = "trained"
        print("done train!")

    def handle_update_writemodel(self, this_client_id, msg):
        logger.debug(f"do handle_update_writemodel")
        state = self.client_dict[this_client_id]["state"]
        if state == "eva_conn_done":
            self.client_dict[this_client_id]["state"] = "model_recv"
            self.send_task("START_TRAIN", this_client_id)  # hmm
            count_model_recv = sum(
                1
                for client_info in self.client_dict.values()
                if client_info["state"] == "model_recv"
            )
            if count_model_recv == self.numDevice:
                print(f"Waiting for training round {self.round} from client...")

    def start_round(self):
        logger.debug(f"do start_round")
        self.round = self.round + 1

        print(f"server start round {self.round}")
        self.round_state = "started"

        # logger.info("1st: Server send task EVACONN")
        logger.warning(f"client_dict in server: {self.client_dict}")
        
        for client_i in self.client_dict:
            self.send_task("EVALUATION_CONNECT", client_i)
        
        logger.warning(f"Check the client_trainres_dict \n {self.client_trainres_dict}")
        while len(self.client_trainres_dict) != self.numDevice:
            time.sleep(1)
        time.sleep(1)
        self.end_round()

    def do_aggregate(self):
        print("Do aggregate ...")
        self.aggregated_models()

    def handle_next_round_duration(self):
        while len(self.client_trainres_dict) < self.numDevice:
            time.sleep(1)

    def end_round(self):
        print(f"server end round {self.round}")

        self.round_state = "finished"

        if self.round < self.numRound:
            self.handle_next_round_duration()
            # logger.warning("doing aggregated")
            self.do_aggregate()
            t = threading.Timer(1, self.start_round)
            t.start()
        else:
            # logger.warning("doing aggregated")
            self.do_aggregate()
            for c in self.client_dict:
                self.send_task("STOP", c)
                print(f"send task STOP {c}")
            self.loop_stop()

    def aggregated_models(self):
        sum_state_dict = OrderedDict()

        for client_id, state_dict in self.client_trainres_dict.items():
            for key, value in state_dict.items():
                if key in sum_state_dict:
                    sum_state_dict[key] = sum_state_dict[key] + torch.tensor(
                        value, dtype=torch.float32
                    )
                else:
                    sum_state_dict[key] = torch.tensor(value, dtype=torch.float32)
        num_models = len(self.client_trainres_dict)
        avg_state_dict = OrderedDict(
            (key, value / num_models) for key, value in sum_state_dict.items()
        )
        torch.save(avg_state_dict, "src/parameters/server.pt")
        self.client_trainres_dict.clear()

server_run = Server(client_fl_id="server")
server_run.connect(host='192.168.100.105', port=1883, keepalive=3600)

server_run.on_connect
server_run.on_disconnect
server_run.on_message
server_run.on_subscribe

server_run.loop_start()

server_run.subscribe(topic="FL/join")

while server_run.numDevice > len(server_run.client_dict):
    time.sleep(1)

server_run.start_round()
server_run._thread.join()

print(f'server exits')