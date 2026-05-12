import time
from pymodbus.client.sync import ModbusTcpClient
from pymodbus.exceptions import ModbusIOException

# ---------------------------------------------
# PLC CONFIGURATION
# ---------------------------------------------
SERVER_IP    = "192.168.3.1"
PORT         = 507
UNIT_ID      = 1
READ_OFFSET  = 1     # 40001
WRITE_OFFSET = 2     # 40002
POLL_SEC     = 0.005
RESET_DELAY  = 0.050


# =============================================
#  PLC CONNECTOR CLASS
# =============================================
class PLCConnector:
    def __init__(self,
                 ip=SERVER_IP,
                 port=PORT,
                 unit=UNIT_ID,
                 read_offset=READ_OFFSET,
                 write_offset=WRITE_OFFSET):

        self.ip = ip
        self.port = port
        self.unit = unit
        self.read_offset = read_offset
        self.write_offset = write_offset

        self.client = None
        self.connect()

    # -----------------------------------------
    # CONNECT TO PLC
    # -----------------------------------------
    def connect(self):
        try:
            self.client = ModbusTcpClient(self.ip, port=self.port)
            if self.client.connect():
                print(f"[PLC] Connected to {self.ip}:{self.port}")
            else:
                print(f"[PLC] Connection failed to {self.ip}:{self.port}")
        except Exception as e:
            print("[PLC] Connection error:", e)

    # -----------------------------------------
    # SAFE RECONNECT
    # -----------------------------------------
    def ensure_connection(self):
        if self.client is None or not self.client.connect():
            print("[PLC] Reconnecting...")
            self.connect()

    # -----------------------------------------
    # READ A HOLDING REGISTER (40001 etc.)
    # -----------------------------------------
    def read_register(self, offset=None):
        self.ensure_connection()

        if offset is None:
            offset = self.read_offset

        try:
            result = self.client.read_holding_registers(offset, 1, unit=self.unit)

            if isinstance(result, ModbusIOException) or not result.isError():
                return result.registers[0]

            print("[PLC] Read error:", result)
            return None

        except Exception as e:
            print("[PLC] Exception during read:", e)
            return None

    # -----------------------------------------
    # WRITE TO A HOLDING REGISTER
    # -----------------------------------------
    def write_register(self, value, offset=None):
        self.ensure_connection()

        if offset is None:
            offset = self.write_offset

        try:
            result = self.client.write_register(offset, value, unit=self.unit)

            if isinstance(result, ModbusIOException) or result.isError():
                print("[PLC] Write error:", result)
            else:
                print(f"[PLC] Wrote {value} → Register {offset}")

        except Exception as e:
            print("[PLC] Exception during write:", e)

    # -----------------------------------------
    # WRITE 1 → WAIT → WRITE 0 (RESET PULSE)
    # -----------------------------------------
    def reset_pulse(self, offset=None, delay=RESET_DELAY):
        if offset is None:
            offset = self.write_offset

        print("[PLC] RESET PULSE")

        self.write_register(1, offset)
        time.sleep(delay)
        self.write_register(0, offset)


# =====================================================
# EXAMPLE USAGE
# =====================================================
if __name__ == "__main__":
    plc = PLCConnector()

    # READ EXAMPLE
    value = plc.read_register()
    print("PLC READ VALUE:", value)

    # WRITE EXAMPLE
    plc.write_register(10)

    # RESET PULSE (write 1 → delay → write 0)
    plc.reset_pulse()
