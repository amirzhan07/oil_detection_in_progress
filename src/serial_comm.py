"""Serial communication utilities for sending navigation commands.

This module defines a simple dataclass for configuring serial ports
and a helper class for sending commands.  It wraps pyserial to
provide context management and automatic cleanup.  If the serial
library is unavailable (for example on systems without an attached
device) the class will raise an informative error at import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import serial  # type: ignore[import]
except ImportError as exc:
    serial = None  # type: ignore[assignment]
    _import_error = exc
else:
    _import_error = None


@dataclass
class SerialConfig:
    """Configuration for opening a serial port."""
    port: str
    baudrate: int = 9600
    timeout: float = 1.0


class SerialCommandSender:
    """Context manager for sending commands over a serial connection."""

    def __init__(self, config: SerialConfig) -> None:
        if serial is None:
            raise ImportError(
                "pyserial is not available.  Install it via pip to use SerialCommandSender"
            ) from _import_error
        self.config = config
        self._ser: Optional[serial.Serial] = None

    def open(self) -> None:
        """Open the serial port using the provided configuration."""
        if self._ser is not None:
            return
        self._ser = serial.Serial(
            port=self.config.port,
            baudrate=self.config.baudrate,
            timeout=self.config.timeout,
        )

    def send(self, command: str) -> None:
        """Send a single command string over the serial connection.

        The command will be encoded as UTF-8 and terminated with a
        newline if one is not already present.
        """
        if self._ser is None:
            self.open()
        # Ensure we always send a newline
        if not command.endswith("\n"):
            command += "\n"
        assert self._ser is not None  # for type checkers
        self._ser.write(command.encode("utf-8"))
        self._ser.flush()

    def close(self) -> None:
        """Close the serial port if it is open."""
        if self._ser is not None:
            self._ser.close()
            self._ser = None

    def __enter__(self) -> 'SerialCommandSender':
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
