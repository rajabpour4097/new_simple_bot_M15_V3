


class BotState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.fib_levels = None
        self.first_touch = False
        self.first_touch_value = None
        self.second_touch = False
        self.second_touch_value = None
        self.fib0_time = None
        self.fib1_time = None
        # self.current_swing = False
