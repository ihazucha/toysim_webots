from datalink.network import TcpClient
from datalink.ipc import SPMCQueue, AddrType
import os
import signal

if __name__ == "__main__":
    q_state = SPMCQueue("q_state", AddrType.TCP, port=10001)
    q_control = SPMCQueue("q_control", AddrType.TCP, port=10002)
    original_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    tcp_client = TcpClient(("localhost", 9999), q_recv=q_control, q_send=q_state)
    tcp_client.start()
    signal.signal(signal.SIGINT, original_handler)

    try:
        tcp_client.join()
    except (KeyboardInterrupt, SystemExit):
        tcp_client.terminate()
    finally:
        tcp_client.join()
        os._exit(0)