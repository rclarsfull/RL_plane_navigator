from stable_baselines3.common.callbacks import BaseCallback
import numbers

class LogAllInfoCallback(BaseCallback):
    def _on_step(self):
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        for info in infos:
            for key, value in info.items():
                if isinstance(value, numbers.Number):
                    self.logger.record(f"env/{key}", value)
                elif isinstance(value, dict):
                    for sub_key, sub_val in value.items():
                        if isinstance(sub_val, numbers.Number):
                            self.logger.record(f"env/{key}/{sub_key}", sub_val)
        return True