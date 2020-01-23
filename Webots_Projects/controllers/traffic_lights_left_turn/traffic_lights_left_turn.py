from controller import Robot
from controller import LED


class TrafficLightsLeftTurnControl(Robot):
    TIME_STEP = 64
    N_LIGHT = 6

    def initialize(self):
        self.red_light = []
        self.orange_light = []
        self.green_light = []

        for i in range(self.N_LIGHT):
            red_light_string = 'red light ' + str(i)
            orange_light_string = 'orange light ' + str(i)
            green_light_string = 'green light ' + str(i)
            self.red_light.append(self.getLED(red_light_string))
            self.orange_light.append(self.getLED(orange_light_string))
            self.green_light.append(self.getLED(green_light_string))

    def run(self):
        basic_time_step = self.getBasicTimeStep()
        while self.step(int(basic_time_step)) != -1:
            self.orange_light[0].set(0)
            self.orange_light[1].set(0)
            self.orange_light[2].set(0)
            self.orange_light[3].set(0)
            self.orange_light[4].set(0)
            self.orange_light[5].set(0)
            self.red_light[0].set(1)
            self.red_light[1].set(1)
            self.red_light[3].set(1)
            self.red_light[4].set(1)
            self.red_light[2].set(0)
            self.red_light[5].set(0)
            self.green_light[0].set(0)
            self.green_light[1].set(0)
            self.green_light[3].set(0)
            self.green_light[4].set(0)
            self.green_light[2].set(1)
            self.green_light[5].set(1)


controller = TrafficLightsLeftTurnControl()
controller.initialize()
controller.run()
