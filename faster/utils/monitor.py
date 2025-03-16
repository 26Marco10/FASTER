import psutil
import numpy as np
from threading import Thread, Event


class CpuMonitor:
    def __init__(self):
        self.cpu_percentages = []
        self.running = False

    def start_cpu_monitoring(self):
        self.cpu_percentages = []
        self.running = True

        def monitor():
            while self.running:
                core_usages = psutil.cpu_percent(interval=0.5, percpu=True)
                self.cpu_percentages.append(sum(core_usages) / len(core_usages))  # Media dei core

        self.thread = Thread(target=monitor)
        self.thread.start()

    def stop_cpu_monitoring(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        # Calcola una media ponderata per dare meno importanza a valori estremi
        weights = np.linspace(1, 0, len(self.cpu_percentages))  # Pesi decrescenti
        return np.average(self.cpu_percentages, weights=weights) if self.cpu_percentages else 0.0
