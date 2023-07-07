"""
Docker container helpers
"""

import socket
from os import system

from VSPEC import config


def is_port_in_use(port: int) -> bool:
    """
    Check if a port is in use on your machine.

    This function is useful to determine if a specific port is already being used
    by another process, such as PSG (Planetary Spectrum Generator).
    It attempts to establish a connection to the specified port on the local machine.
    If the connection is successful (return value of 0), it means that something is
    already running on that port.

    Parameters
    ----------
    port : int
        The port number to check. Typically, PSG runs on port 3000.

    Returns
    -------
    bool
        Returns True if a process is already running on the specified port, and False otherwise.

    Notes
    -----
    - If you call this function immediately after changing the Docker image state,
        you may get an incorrect answer due to timing issues. It is recommended to use
        this function within a function that incorporates a timeout mechanism.
    - This function relies on the `socket` module from the Python standard library.

    Examples
    --------
    >>> is_port_in_use(3000)
    True

    >>> is_port_in_use(8080)
    False

    """
    socket_obj = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    return socket_obj.connect_ex(('localhost', port)) == 0


def set_psg_state(running: bool):
    """
    Set the local PSG (Planetary Spectrum Generator) state.

    This function allows you to control the state of the local
    PSG Docker container. By specifying whether the PSG should be running
    or not, you can start or stop the PSG container accordingly.

    Parameters
    ----------
    running : bool
        A boolean value indicating whether the PSG should be running.
        - If `running` is True and the PSG is not already running, the function will start the PSG container.
        - If `running` is False and the PSG is currently running, the function will stop the PSG container.

    Notes
    -----
    - This function relies on the `system` function from the `os` module to execute Docker commands.
    - The `is_port_in_use` function from the `VSPEC.helpers` module is used to check if the PSG port is in use.

    Examples
    --------
    >>> set_psg_state(True)  # Start the PSG container if not already running

    >>> set_psg_state(False)  # Stop the PSG container if currently running
    """
    if is_port_in_use(config.PSG_PORT) and not running:
        system('docker stop psg')
    elif not is_port_in_use(config.PSG_PORT) and running:
        system('docker start psg')
