"""W&B implementation replacing Neptune API with silly names to match.

"""

import os

class NullFloatSeries:

    def fetch_value(self):
        raise AttributeError("fetch_value not implemented for NullFloatSeries")

    def append(self, value, step=None):
        pass


class NullRun:

    def __getitem__(self, key):
        return NullFloatSeries()

    def __setitem__(self, key, value):
        pass

    def stop(self):
        pass

    def get_run_id(self) -> str:
        return "NoIDset"


class WandbFloatSeries:
    """Captures a metric key and logs to W&B when append() is called."""

    def __init__(self, key: str):
        self.key = key

    def append(self, value, step=None):
        import wandb
        if "vs_pflops" in self.key:
            wandb.log({self.key: value, "pflops": step})
        elif "vs_num_tokens" in self.key:
            wandb.log({self.key: value, "num_tokens": step})
        else:
            wandb.log({self.key: value})#, step=int(step) if step is not None else None)


class NeptuneRunWrapper:
    """Wrap W&B API with the same interface as the old Neptune wrapper."""

    def __init__(
        self,
        use_neptune: bool,  # kept as use_neptune for minimal changes to callers
        description: str,
        run_name: str | None = None,
        print_calls: bool = False,
        tags: list[str] = [],
    ):
        self.print_calls = print_calls
        self._use_wandb = use_neptune  # use_neptune flag now means use_wandb
        if use_neptune:
            import wandb
            print("Using W&B")
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "training-exploration"),
                entity=os.environ.get("WANDB_ENTITY", None),
                name=run_name,
                notes=description,
                tags=tags,
            )
            # custom x axes
            wandb.define_metric("loss_vs_pflops/*", step_metric="pflops")
            wandb.define_metric("loss_vs_num_tokens/*", step_metric="num_tokens")      
            self.run = wandb.run
        else:
            self.run = NullRun()

    def get_run_id(self) -> str:
        if self._use_wandb:
            import wandb
            return wandb.run.id
        return self.run.get_run_id()

    def __getitem__(self, key):
        if self.print_calls:
            print(f"W&B getitem: {key}")
        if self._use_wandb:
            return WandbFloatSeries(key)
        return self.run[key]

    def __setitem__(self, key, value):
        if self.print_calls:
            print(f"W&B setitem: {key}={value}")
        if self._use_wandb:
            import wandb
            value = self._convert_value(value)
            wandb.run.summary[key] = value
        else:
            self.run[key] = value

    def _convert_value(self, value):
        if isinstance(value, dict):
            return {k: self._convert_value(v) for k, v in value.items()}
        if value is None or isinstance(value, (list, tuple, set)):
            return str(value)
        return value

    def stop(self):
        if self.print_calls:
            print("W&B stop")
        if self._use_wandb:
            import wandb
            wandb.finish()
        else:
            self.run.stop()