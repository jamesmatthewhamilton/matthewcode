"""Consecutive-identical tool-call loop detector."""

import hashlib
import json


class LoopDetector:
    """Trips when the same (name, args, result) signature appears `threshold`
    times consecutively. Any different signature resets the counter.

    Hashing the result as part of the signature means legitimate edit-build-test
    cycles (different result → different signature) do not trip.
    """

    def __init__(self, threshold: int = 3, enabled: bool = True):
        self.threshold = threshold
        self.enabled = enabled
        self._last_sig = None
        self._count = 0

    def record(self, name: str, args: dict, result: str) -> bool:
        """Return True iff this call trips the loop threshold."""
        if not self.enabled:
            return False
        sig = self._signature(name, args, result)
        if sig == self._last_sig:
            self._count += 1
        else:
            self._last_sig = sig
            self._count = 1
        return self._count >= self.threshold

    def reset(self) -> None:
        self._last_sig = None
        self._count = 0

    @staticmethod
    def _signature(name: str, args: dict, result: str) -> str:
        args_json = json.dumps(args, sort_keys=True, default=str)
        result_hash = hashlib.sha256(result.encode("utf-8", "replace")).hexdigest()
        payload = f"{name}\0{args_json}\0{result_hash}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


LOOP_WARNING = (
    "\n\n[LOOP DETECTED] You have called `{name}` {count} times in a row with "
    "identical arguments and identical results. DO NOT call this tool again "
    "with these arguments. Try a different approach, or ask the user for guidance."
)
