import pyopencl as cl
from typing import *

def list_devices() -> List[Tuple[cl.Platform,cl.Device]]:
  devices_list = []
  Id = 0
  for plataform in cl.get_platforms():
    for device in plataform.get_devices():
      print(f"ID: {Id} | {plataform.vendor.lstrip().rstrip()} | {device.name.lstrip().rstrip()}")
      devices_list.append((plataform,device))
  return devices_list

def show_device_info(device: cl.Device) -> None:
    device_type = "Unknown"
    if device.type & cl.device_type.CPU:
        device_type = "CPU"
    elif device.type & cl.device_type.GPU:
        device_type = "GPU"
    elif device.type & cl.device_type.ACCELERATOR:
        device_type = "Accelerator"

    print("======Device Information============================")
    print(f"Name: {device.name.strip()}")
    print(f"Vendor: {device.vendor.strip()}")
    print(f"Type: {device_type}")
    print(f"Version: {device.version.strip()}")
    print(f"OpenCL C Version: {device.opencl_c_version.strip()}")
    print("====================================================")
    print(f"Global Memory Size: {device.global_mem_size / (1024 ** 2):.2f} MB")
    print(f"Local Memory Size: {device.local_mem_size / 1024:.2f} KB")
    print(f"Max Constant Buffer Size: {device.max_constant_buffer_size / 1024:.2f} KB")
    print("====================================================")
    print(f"Max Compute Units: {device.max_compute_units}")
    print(f"Max Clock Frequency: {device.max_clock_frequency} MHz")
    print("====================================================")
    print(f"Max Work Group Size: {device.max_work_group_size}")
    print(f"Max Work Item Dimensions: {device.max_work_item_dimensions}")
    print(f"Max Work Item Sizes: {device.max_work_item_sizes}")
    print("====================================================")
    print(f"Preferred Vector Width (float): {device.preferred_vector_width_float}")
    print(f"Preferred Vector Width (double): {device.preferred_vector_width_double}")
    print("====================================================")
    double_fp = device.get_info(cl.device_info.DOUBLE_FP_CONFIG)
    print(f"Double Precision Support: {'Yes' if double_fp != 0 else 'No'}")

if __name__ == "__main__":
   for (plataform,device) in list_devices():
      show_device_info(device)