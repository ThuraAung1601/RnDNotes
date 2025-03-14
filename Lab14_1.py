import time
import threading


class TrafficLight:
    def __init__(self):
        self.current_color = None
        self.target_color = "red"
        self.countdown_time = 0
        self.running = False

    def start(self):
        self.running = True
        self.change_color("red")  # Start with red light

    def stop(self):
        self.running = False

    def tick(self):
        while self.running:
            if self.countdown_time > 0:
                print(f"[{self.current_color.upper()}] Changing to {self.target_color.upper()} in {self.countdown_time} seconds")
                self.countdown_time -= 1
            else:
                self.change_color(self.target_color)
            time.sleep(1)

    def set_timer(self, duration):
        self.countdown_time = duration

    def change_color(self, color):
        self.current_color = color
        if color == "red":
            self.target_color = "green"
            self.set_timer(4)
            print("[RED] Light is RED for 4 seconds")
        elif color == "green":
            self.target_color = "yellow"
            self.set_timer(3)
            print("[GREEN] Light is GREEN for 3 seconds")
        elif color == "yellow":
            self.target_color = "red"
            self.set_timer(2)
            print("[YELLOW] Light is YELLOW for 2 seconds")

    def run(self):
        thread = threading.Thread(target=self.tick)
        thread.start()


if __name__ == "__main__":
    traffic_light = TrafficLight()
    traffic_light.start()
    traffic_light.run()
