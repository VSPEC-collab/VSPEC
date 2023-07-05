
from time import time
from os import system

from VSPEC import helpers


def port_is_running(port: int, timeout: float, target_state: bool):
    """
    It takes some time to know if a port has been switched off.
    We tell this function what we expect, and it listens for it.
    Say we turn PSG off, it asks "Is PSG running?" for 10 seconds
    until it hears back `False`. If it never does, it tells us that
    PSG is still running.

    Parameters
    ----------
    port : int
        The port to listen on.
    timeout : float
        The length of time to listen for in seconds.
    target_state : bool
        The expected state at the end of listening.

    Returns
    -------
    psg_state : bool
        The state of PSG. `True` if running on port `port`, else `False`.
    """
    timeout_time = time() + timeout
    while True:
        if target_state == True:
            if helpers.is_port_in_use(port):
                return True
            elif time() > timeout_time:
                return False
        else:
            if not helpers.is_port_in_use(port):
                return False
            elif time() > timeout_time:
                return True


def test_is_port_in_use():
    """
    Test `VSPEC.is_port_in_use`
    """
    default_psg_port = 3000
    timeout_duration = 10  # timeout after 10s
    previous_state = helpers.is_port_in_use(default_psg_port)
    system('docker stop psg')
    system('docker start psg')
    if not port_is_running(default_psg_port, timeout_duration, True):
        raise RuntimeError('Test failed -- timeout')
    system('docker stop psg')
    if port_is_running(default_psg_port, timeout_duration, False):
        raise RuntimeError('Test failed -- timeout')
    helpers.set_psg_state(previous_state)


def test_set_psg_state():
    """
    Test `VSPEC.helpers.set_psg_state`
    """
    psg_port = 3000
    timeout = 20
    previous_state = helpers.is_port_in_use(psg_port)
    helpers.set_psg_state(True)
    assert port_is_running(psg_port, timeout, True)
    helpers.set_psg_state(True)
    assert port_is_running(psg_port, timeout, True)
    helpers.set_psg_state(False)
    assert not port_is_running(psg_port, timeout, False)
    helpers.set_psg_state(False)
    assert not port_is_running(psg_port, timeout, False)
    helpers.set_psg_state(True)
    assert port_is_running(psg_port, timeout, True)
    helpers.set_psg_state(previous_state)
