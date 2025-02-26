import time


def format_time(seconds):
    units = [
        ("hours", 3600),
        ("minutes", 60),
        ("seconds", 1),
        ("milliseconds", 1e-3),
        ("microseconds", 1e-6),
    ]

    for unit_name, unit_value in units:
        if seconds >= unit_value:
            time_value = seconds / unit_value
            return f"{time_value:.2f} {unit_name}"
    return "0 seconds"  # Handle case for 0 seconds


class Profiler:
    def __init__(self):
        self.timers = {}
        self.counters = {}

    def create_timer(self, name):
        self.timers[name] = {
            "start_time": None,
            "elapsed_time": 0,
            "running": False,
        }

    def start_timer(self, name):
        if name not in self.timers:
            raise ValueError(f"Timer '{name}' does not exist.")
        if self.timers[name]["running"]:
            raise ValueError(f"Timer '{name}' is already running.")
        self.timers[name]["start_time"] = time.time()
        self.timers[name]["running"] = True

    def pause_timer(self, name):
        if name not in self.timers:
            raise ValueError(f"Timer '{name}' does not exist.")
        if not self.timers[name]["running"]:
            raise ValueError(f"Timer '{name}' is not running.")
        self.timers[name]["elapsed_time"] += time.time() - self.timers[name]["start_time"]
        self.timers[name]["running"] = False

    def get_timer_sec(self, name):
        if name not in self.timers:
            raise ValueError(f"Timer '{name}' does not exist.")
        if self.timers[name]["running"]:
            current_time = self.timers[name]["elapsed_time"] + (time.time() - self.timers[name]["start_time"])
        else:
            current_time = self.timers[name]["elapsed_time"]
        return current_time

    def get_all_timers(self):
        all_timers = {}
        for name in self.timers:
            all_timers[name] = self.get_timer_sec(name)
        return all_timers

    def report_timer_string(self, name):
        return f"{name} elapsed time: {format_time(self.get_timer_sec(name))}"

    def create_and_start_timer(self, name):
        self.create_timer(name)
        self.start_timer(name)


    # Counter
    def inc(self,key:str,delta:int=1):
        self.counters[key] = self.counters.get(key,0) + delta

    def set_counter(self,key:str,to=0):
        self.counters[key] = to

    def get_counter(self,key:str):
        return self.counters.get(key,0)
