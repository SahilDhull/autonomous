"""Defines Functions to check existence of GPU in the system.
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""


def check_system_gpu():
    """This function is intended to automatically check existence of CUDA-enabled GPU and return its device ID."""

    # The following seems to be system dependent and is disabled for now.
    # The user should manually set if their system has GPU or not.

    # try:
    #     import GPUtil
    #     import os
    #
    #     # Set CUDA_DEVICE_ORDER so the IDs assigned by CUDA match those from nvidia-smi
    #     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #
    #     # Get the first available GPU
    #     device_id_list = GPUtil.getFirstAvailable()
    #     gpu_cpu_id = device_id_list[0]  # grab first element from list
    #     has_gpu = True
    # except Exception as gpu_ex:
    #     print("GPU Search Error: {}  ! Will use CPU for SqueezeDet.".format(repr(gpu_ex)))
    #     has_gpu = False
    #     gpu_cpu_id = 0

    has_gpu = False
    gpu_cpu_id = 0
    if not has_gpu:
        print("System does not have a CUDA-enabled GPU. I will use CPU for SqueezeDet.")
    return has_gpu, gpu_cpu_id
