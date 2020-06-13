import torch


def get_device():
    """
    Select GPU if available, else CPU.

    :return: device variable used for further training
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return device
