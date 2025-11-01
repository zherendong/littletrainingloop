"""Null implementation for neptune API.

Meant to allow running experiments without reporting and simplify the code that reports to neptune.
"""

import os


class NullNeptuneFloatSeries:

    def fetch_value(self):
        raise NotImplementedError()

    def append(self, value, step=None):
        pass


class NullNeptuneRun:

    def __getitem__(self, key):
        return NullNeptuneFloatSeries()

    def __setitem__(self, key, value):
        pass

    def stop(self):
        pass


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
        """Convert unsupported Neptune types to supported ones."""
        if value is None:
            return "None"
        elif isinstance(value, tuple):
            return list(value)
        elif isinstance(value, dict):
            # Recursively convert dict values
            return {k: self._convert_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            # Recursively convert list items
            return [self._convert_value(item) for item in value]
        else:
            return value

    def stop(self):
        if self.print_calls:
            print("Neptune stop")
        self.run.stop()
