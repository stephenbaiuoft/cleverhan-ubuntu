from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def main():
  g = get_available_gpus()
  rez = ";".join(g)
  #print("gpu probablity not matched: >" + rez  + "<")

main()