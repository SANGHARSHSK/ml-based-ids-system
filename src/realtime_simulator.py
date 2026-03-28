import numpy as np
import pandas as pd
import time
import threading
import queue
from datetime import datetime

class RealTimeSimulator:
    """
    Simulates real-time network traffic
    Feeds packets to model one by one
    Shows live detection in dashboard
    """
    def __init__(self, model, scaler, features,
                 X_test, y_test):
        self.model    = model
        self.scaler   = scaler
        self.features = features
        self.X_test   = X_test
        self.y_test   = y_test
        self.queue    = queue.Queue()
        self.running  = False
        self.logs     = []

    def start_simulation(self, speed=0.5):
        """Start feeding packets in background thread"""
        self.running = True
        thread = threading.Thread(
            target=self._simulate_packets,
            args=(speed,),
            daemon=True
        )
        thread.start()
        print("✅ Real-time simulation started!")

    def stop_simulation(self):
        self.running = False
        print("⏹️ Simulation stopped.")

    def _simulate_packets(self, speed):
        """Feed test data as simulated packets"""
        idx = 0
        while self.running and idx < len(self.X_test):
            packet    = self.X_test[idx]
            true_label = self.y_test[idx]

            # Predict
            pred  = self.model.rf.predict([packet])[0]
            prob  = self.model.rf.predict_proba([packet])[0][1]

            result = {
                'timestamp'  : datetime.now().strftime(
                                "%H:%M:%S"),
                'packet_id'  : idx,
                'prediction' : pred,
                'probability': round(float(prob), 4),
                'true_label' : int(true_label),
                'status'     : '⚠️ ATTACK'
                                if pred == 1
                                else '✅ NORMAL',
                'correct'    : pred == true_label
            }

            self.queue.put(result)
            self.logs.append(result)

            # Keep only last 1000 logs (RAM safe)
            if len(self.logs) > 1000:
                self.logs = self.logs[-1000:]

            idx  += 1
            time.sleep(speed)   # control speed

    def get_latest(self, n=10):
        """Get latest n detections"""
        return self.logs[-n:]

    def get_stats(self):
        """Live statistics"""
        if not self.logs:
            return {}
        total   = len(self.logs)
        attacks = sum(1 for l in self.logs
                      if l['prediction'] == 1)
        correct = sum(1 for l in self.logs
                      if l['correct'])
        return {
            'total'         : total,
            'attacks'       : attacks,
            'normal'        : total - attacks,
            'accuracy'      : round(correct/total*100, 1),
            'attack_rate'   : round(attacks/total*100, 1)
        }