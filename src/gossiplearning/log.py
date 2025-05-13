from pathlib import Path

from gossiplearning.config import LogLevel
from gossiplearning.models import Time, NodeId


class Logger:
    """Logger utility for class for gossip learning simulator."""

    def __init__(self, log_level: LogLevel, workspace_dir: Path):
        """Initialize the logger."""
        self._level = log_level
        self._log_file = workspace_dir / "logs.txt"

        # clear log file content
        open(self._log_file, "w").close()

    def _add_log_line(self, text: str):
        """
        Append a new line to the log file.

        :param text: the line text
        """
        with self._log_file.open("a") as log_file:
            log_file.write(f"{text}\n")

    def node_event_log(self, msg: str, time: Time, node: NodeId):
        """
        Add a new INFO log entry representing a single node event.

        :param msg: the message to be logged
        :param time: the simulated time when the event happened
        :param node: the ID of the node where the event happened
        """
        #         print(colored(f"[ {time}s : N{node} ]", "blue", attrs=["bold"]), msg)
        if self._level >= LogLevel.INFO:
            self._add_log_line(f"[INFO]  [{time}s] - [Node {node}] : {msg}")

    def debug_log(self, msg: str):
        """
        Add a new DEBUG log entry.

        :param msg: the message to be logged
        """
        if self._level >= LogLevel.DEBUG:
            self._add_log_line(f"[DEBUG]: {msg}")
