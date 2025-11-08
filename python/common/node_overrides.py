
from typing import Tuple, Any, Dict, Optional

from common.lucid_vision import TAB1, TAB2

def _resolve_node(nodemap, node_name: str):
    """
    Resolve a node from either a device nodemap (get_node) or a TL stream nodemap (mapping access).
    Returns the node object or None if not found.
    """
    node = None
    # Try Arena device nodemap style
    try:
        get_node = getattr(nodemap, "get_node", None)
        if callable(get_node):
            node = get_node(node_name)
    except Exception:
        node = None
    # Try mapping-style access (TL stream nodemap)
    if node is None:
        try:
            node = nodemap[node_name]  # type: ignore[index]
        except Exception:
            node = None
    return node


def safe_get_node_value(nodemap, node_name: str, default: Any = None) -> Any:
    """
    Safely get a node's value; return default if node is missing or unreadable.
    """
    node = _resolve_node(nodemap, node_name)
    if node is None:
        return default
    try:
        return node.value
    except Exception:
        return default


def safe_set_node(nodemap, node_name: str, value: any, strict: bool = False) -> bool:
    """
    Safely set a node's value. Returns True if set, False if skipped.
    If strict is True, raise on errors; otherwise print a concise warning and continue.
    """
    node = _resolve_node(nodemap, node_name)
    if node is None:
        if strict:
            raise AttributeError(f"Node '{node_name}' not found")
        print(f"{TAB1}Warning: Node '{node_name}' not found; skipping")
        return False
    try:
        node.value = value
        return True
    except Exception as e:
        if strict:
            raise
        print(f"{TAB1}Warning: Could not set '{node_name}' -> {value}: {e}")
        return False


class TemporaryNodes:
    """
    Context manager that temporarily overrides node values on a nodemap and restores them on exit.

    Example:
        with TemporaryNodes(device.nodemap, {
            'PixelFormat': PixelFormat.Mono8,
            'AcquisitionMode': 'Continuous',
        }):
            device.start_stream()
            ...
    """
    def __init__(
        self,
        nodemap,
        assignments: Dict[str, Any],
        strict: bool = False,
        commit: bool = False,
    ) -> None:
        self._nodemap = nodemap
        self._assignments = assignments or {}
        self._strict = strict
        self._commit = commit
        self._original_values: Dict[str, Any] = {}
        self._applied_keys: Dict[str, bool] = {}

    def __enter__(self):
        for name, new_value in self._assignments.items():
            node = _resolve_node(self._nodemap, name)
            if node is None:
                if self._strict:
                    raise AttributeError(f"Node '{name}' not found")
                print(f"{TAB1}Warning: Node '{name}' not found; skipping override")
                continue
            # Save original value
            try:
                self._original_values[name] = node.value
            except Exception:
                # If value cannot be read, still attempt to set and skip restore
                self._original_values[name] = None
            # Apply new value
            applied = safe_set_node(self._nodemap, name, new_value, strict=self._strict)
            self._applied_keys[name] = applied
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._commit:
            return False
        # Restore in reverse order of application to be conservative
        for name in list(self._original_values.keys())[::-1]:
            if not self._applied_keys.get(name, False):
                continue
            original_value = self._original_values[name]
            if original_value is None:
                # No readable original; skip restore
                continue
            try:
                safe_set_node(self._nodemap, name, original_value, strict=False)
            except Exception:
                # Best-effort restore
                pass
        return False


class CameraOverrides:
    """
    Convenience context to manage both device nodemap and TL stream node overrides.

    Example:
        with CameraOverrides(device,
                             nodes={'PixelFormat': PixelFormat.Mono8},
                             stream_nodes={'StreamBufferHandlingMode': 'NewestOnly'}):
            device.start_stream()
            ...
    """
    def __init__(
        self,
        device,
        nodes: Optional[Dict[str, Any]] = None,
        stream_nodes: Optional[Dict[str, Any]] = None,
        strict: bool = False,
        commit: bool = False,
    ) -> None:
        self._device = device
        self._strict = strict
        self._commit = commit
        self._ctx_device: Optional[TemporaryNodes] = None
        self._ctx_stream: Optional[TemporaryNodes] = None
        self._nodes = nodes or {}
        self._stream_nodes = stream_nodes or {}

    def __enter__(self):
        if self._nodes:
            self._ctx_device = TemporaryNodes(self._device.nodemap, self._nodes, strict=self._strict, commit=self._commit)
            self._ctx_device.__enter__()
        if self._stream_nodes:
            self._ctx_stream = TemporaryNodes(self._device.tl_stream_nodemap, self._stream_nodes, strict=self._strict, commit=self._commit)
            self._ctx_stream.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        # Exit in reverse creation order
        if self._ctx_stream is not None:
            try:
                self._ctx_stream.__exit__(exc_type, exc, tb)
            finally:
                self._ctx_stream = None
        if self._ctx_device is not None:
            try:
                self._ctx_device.__exit__(exc_type, exc, tb)
            finally:
                self._ctx_device = None
        return False
