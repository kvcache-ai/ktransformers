import re
import itertools
import time
import enum
import math
from enum import StrEnum

class ProfStatKey(StrEnum):
    ExpertsSummitCurrLayer = "ExpertsSummitCurrLayer"
    ExpertsSummitNextLayer = "ExpertsSummitNextLayer"
    ExpertsCPUForwardOne = "ExpertsCPUForwardOne"
    ExpertsCPUForwardTwo = "ExpertsCPUForwardTwo"
    CPUMoEKExpertsCallback = "CPUMoEKExpertsCallback"

class ProfTimeStat:
    def __init__(self):
        # open_status = os.environ["KT_PERF_STAT"] if "KT_PERF_STAT" in os.environ else "0"
        # if open_status == "0":
        #     self.on = False
        # else:
        #     self.on = True
        self.on = False
        self.prefill_stats = dict()
        self.decode_stats = dict()
        for key in ProfStatKey:
            self.prefill_stats[key] = ProfStatItem()
            self.decode_stats[key] = ProfStatItem()
        self.reset_all()

    def record_start_time(self):
        start_time = time.time_ns()
        return start_time

    def add_time_stat(self, key: ProfStatKey, time_ns, is_prefill):
        if not key:
            return
        # torch.cuda.synchronize()
        cost = time.time_ns() - time_ns
        if is_prefill:
            item = self.prefill_stats[key]
        else:
            item = self.decode_stats[key]
        item.add_item(cost)

    def print_all(self):
        # rank = f"[rank:{torch.distributed.get_rank()}]"
        rank = f"[rank:0]"
        msg = f"\n{rank} Prefill Time Stat\n"
        msg += rank + " {:27}{:>15}{:>15}{:>15}{:>15}{:>15}{:>15}{:>15}\n".format("", "min(ms)", "max(ms)", "avg(ms)", "count", "total(ms)", ">2ms", ">10ms")
        for key, value in self.prefill_stats.items():
            msg += rank + f" {key.value:<25}:{value.get_stat()}\n"
        msg += f"\n{rank} Decode Time Stat\n"
        msg += rank + " {:27}{:>15}{:>15}{:>15}{:>15}{:>15}{:>15}{:>15}\n".format("", "min(ms)", "max(ms)", "avg(ms)", "count", "total(ms)", ">2ms", ">10ms")
        for key, value in self.decode_stats.items():
            msg += rank + f" {key.value:<25}:{value.get_stat()}\n"
        print(msg)

    def reset_all(self):
        for _, value in self.prefill_stats.items():
            value.reset()
        for _, value in self.decode_stats.items():
            value.reset()


class ProfStatItem:
    def __init__(self):
        self.min_time = 100000000
        self.max_time = 0
        self.total_time_ns = 0
        self.count = 0
        self.err_time = []
        self.ms_count2 = 0
        self.ms_count10 = 0

    def add_item(self, cost_time_ns):
        self.count += 1
        self.total_time_ns += cost_time_ns
        self.min_time = min(self.min_time, cost_time_ns)
        self.max_time = max(self.max_time, cost_time_ns)
        if (cost_time_ns > 2000000):
        #   self.err_time.append(round(cost_time_ns / 1000 / 1000, 2))
          self.ms_count2 += 1
        if (cost_time_ns > 10000000):
        #   self.err_time.append(round(cost_time_ns / 1000 / 1000, 2))
          self.ms_count10 += 1
        # self.err_time.append(round(cost_time_ns / 1000 / 1000, 2))

    def reset(self):
        self.min_time = 100000000
        self.max_time = 0
        self.total_time_ns = 0
        self.count = 0

    def get_stat(self):
        min_time = self.min_time / 1000 / 1000
        max_time = self.max_time / 1000 / 1000
        if self.count != 0:
            avg_time = self.total_time_ns / self.count / 1000 / 1000
        else:
            avg_time = 0
        total = self.total_time_ns / 1000 / 1000
        # tmpstr = str(self.err_time)
        # print(f"\r\n err_time: {tmpstr} \r\n ")
        return f"{min_time:15.2f}{max_time:15.2f}{avg_time:15.2f}{self.count:15}{total:15.2f}{self.ms_count2:>15}{self.ms_count10:>15}"


PROF_TIME_STAT = ProfTimeStat()

