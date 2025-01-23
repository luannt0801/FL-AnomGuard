Fed

#### Install Mosquitto Esclipse in Ubuntu

```bash
sudo apt-add-repository ppa:mosquitto-dev/mosquitto-ppa

sudo apt-get update

sudo apt-get install mosquitto

sudo apt-get install mosquitto-clients

sudo apt clean
```

Cấu hình lại file config trong `/etc/mosquitto/mosquitto.conf` bằng cách thêm 2 trường sau:

```bash
listener 1883

allow_anonymous true
```


