"""Null implementation for neptune API.

Meant to allow running experiments without reporting and simplify the code that reports to neptune.
"""

import os


class NullNeptuneFloatSeries:

    def fetch_value(self):
        raise AttributeError("fetch_value not implemented for NullNeptuneFloatSeries")

    def append(self, value, step=None):
        pass


class NullNeptuneRun:

    def __getitem__(self, key):
        return NullNeptuneFloatSeries()

    def __setitem__(self, key, value):
        pass

    def stop(self):
        pass

    def get_run_id(self) -> str:
        return "NoIDset"


class NeptuneRunWrapper:
    """Wrap neptune API to log all calls and allow us to disable it."""

    def __init__(
        self,
        use_neptune: bool,
        description: str,
        run_name: str | None = None,
        print_calls: bool = False,
        tags: list[str] = [],
    ):
        self.print_calls = print_calls
        if use_neptune:
            import neptune
            from dotenv import load_dotenv

            print("Using neptune")
            load_dotenv(dotenv_path=os.path.expanduser(".env"))
            neptune_api_token = os.environ["NEPTUNE_API_TOKEN"]

            self.run = neptune.init_run(
                project="markusrabeworkspace/training-exploration",
                api_token=neptune_api_token,
                description=description or run_name,
                name=run_name,
                source_files="*.py",
                tags=tags,
            )
        else:
            self.run = NullNeptuneRun()

    def get_run_id(self) -> str:
        try:
            return str(self.run["sys/id"].fetch())
        except AttributeError as e:
            print(f"Error getting run id: {e}")
            return "ID_error"

    def __getitem__(self, key):
        if self.print_calls:
            print(f"Neptune getitem: {key}")
        return self.run[key]

    def __setitem__(self, key, value):
        if self.print_calls:
            print(f"Neptune setitem: {key}={value}")
        # Convert unsupported types to Neptune-compatible formats
        value = self._convert_value(value)
        self.run[key] = value

    def _convert_value(self, value):
        """Convert unsupported Neptune types to Neptune-supported values.

        Strategy:
        - Keep dicts as dicts but recursively convert their values
        - Stringify None, lists, tuples, and sets using Neptune's stringify_unsupported
          (falls back to str() if Neptune is not installed)
        - Leave supported scalar types (str, int, float, bool) as-is
        """
        # Dict: recursively sanitize values while preserving structure
        if isinstance(value, dict):
            return {k: self._convert_value(v) for k, v in value.items()}

        # None and common collections (not directly supported by Neptune)
        if value is None or isinstance(value, (list, tuple, set)):
            return self._stringify(value)

        # Pass through supported scalar types and anything else
        return value

    def _stringify(self, obj):
        """Return a Neptune-compatible stringified wrapper if available.

        Tries to use neptune.utils.stringify_unsupported which returns a
        StringifyValue understood by Neptune. If Neptune isn't installed
        (e.g., local runs with use_neptune=False), fall back to str(obj).
        """
        try:
            from neptune.utils import stringify_unsupported  # type: ignore

            return stringify_unsupported(obj)
        except Exception:
            return str(obj)

    def stop(self):
        if self.print_calls:
            print("Neptune stop")
        self.run.stop()
