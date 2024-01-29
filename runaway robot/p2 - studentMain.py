# ----------
# Part Two
#
# Now we'll make the scenario a bit more realistic. Now Traxbot's
# sensor measurements are a bit noisy (though its motions are still
# completetly noise-free and it still moves in an almost-circle).
# You'll have to write a function that takes as input the next
# noisy (x, y) sensor measurement and outputs the best guess 
# for the robot's next position.
#
# ----------
# YOUR JOB
#
# Complete the function estimate_next_pos. You will be considered 
# correct if your estimate is within 0.01 stepsizes of Traxbot's next
# true position. 
#
# ----------
# GRADING
# 
# We will make repeated calls to your estimate_next_pos function. After
# each call, we will compare your estimated position to the robot's true
# position. As soon as you are within 0.01 stepsizes of the true position,
# you will be marked correct and we will tell you how many steps it took
# before your function successfully located the target bot.

# These import steps give you access to libraries which you may (or may
# not) want to use.
from robot import *  # Check the robot.py tab to see how this works.
from matrix import * # Check the matrix.py tab to see how this works.
from math import *

class KalmanFilter(object):
    """
    The CircularMotionKalmanFilter class extended the original KalmanFilter by
    implementing a nonlinear transition model and focusing on using m = 2D
    position measurements to estimate n = 5D variables, including:
        1. x position of the robot
        2. y position of the robot
        3. the robot's current heading direction
        4. the robot's turning angle between each measurements
        5. the distance traveled between each measurements
    Since the five variables to estimate do not form a linear dynamic system,
    the transition matrix F is defined to be the Jacobian of the transition
    model.
    The estimation update step is the same as the original Kalman Filter since
    the observation model H remains linear.
    """
    def __init__(self, cov=1000.0, measurementNoise=1.0):
        """ Constructor
        Initializes an Extended Kalman Filter with n=5 variable dimension and
        m=2 measurement dimension, then set up the H and R matrices.
        """
        self.n = 5
        self.m = 2
        self.x = matrix([[]])
        self.x.zero(self.n, 1)
        self.P = matrix([
            [cov, 0, 0, 0, 0],
            [0, cov, 0, 0, 0],
            [0, 0, cov, 0, 0],
            [0, 0, 0, cov, 0],
            [0, 0, 0, 0, cov]
        ])
        self.F = matrix([[]])
        self.F.zero(self.n, self.n)
        self.u = matrix([[]])
        self.u.zero(self.n, 1)
        self.H = matrix([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0]
        ])
        self.R = matrix([
            [measurementNoise, 0],
            [0, measurementNoise]
        ])
        self.I = matrix([[]])
        self.I.identity(self.n)

    def updateMeasurement(self, z):
        """ Update estimations of the EKF based on measurements
        With the input measurements z, updates the x and P matrices.

        Args:
            z: mx1 matrix, the new measurements
        """
        if len(z.value) != self.m or len(z.value[0]) != 1:
            print('Invalid dimensions for z, should be (%d, 1)' % self.m)
            return
        y = z - self.H * self.x
        S = self.H * self.P * self.H.transpose() + self.R
        K = self.P * self.H.transpose() * S.inverse()
        self.x = self.x + K * y
        self.P = (self.I - K * self.H) * self.P

    def predict(self):
        """ Update estimations using Kalman Filter as predictions
        Advnaces in time step and predict the next x and P. Use the current
        estimations to perform the transition model, and use its Jacobian to
        update the covariance matrix P.

        Returns:
            x, y: the predicted x and y positions at the next timestep
        """
        F = self.getTransitionMatrix()
        self.x = self.performTransitionModel()
        self.P = F * self.P * F.transpose()
        return self.x.value[0][0], self.x.value[1][0]

    def performTransitionModel(self):
        """
        Performs the transition model to get the next state of the variables.
        Since we assume a circular motion, x and y positions move by the
        estimated distance along the estimated direction.

        Returns:
            x: nx1 matrix, estimations of the next state
        """
        x = self.getX()
        y = self.getY()
        direction = self.getDirection()
        turnAngle = self.getTurningAngle()
        dist = self.getDistance()
        return matrix([
            [x + dist * cos(direction + turnAngle)],
            [y + dist * sin(direction + turnAngle)],
            [direction + turnAngle],
            [turnAngle],
            [dist]
        ])

    def getTransitionMatrix(self):
        """
        Generate the transition matrix based on the current estimations. Since
        the transition model is nonlinear, the transition matrix would be the
        Jacobian of the transition model. Differentiating the model against the
        variables gives us the matrix as shown below.

        Returns:
            F: nxn matrix, transition matrix / Jacobian of the transition model
        """
        d = self.getDistance()
        angle = self.getDirection() + self.getTurningAngle()
        return matrix([
            [1, 0, -d*sin(angle), -d*sin(angle), cos(angle)],
            [0, 1, d*cos(angle), d*cos(angle), sin(angle)],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]
        ])

    def predictAfterN(self, n):
        """
        Predicts the state of the variables based on the current estimations
        after n time steps, and return the x and y positions.

        Returns:
            x and y positions estimated to be after n time steps
        """
        x = self.performTransitionModel()
        for _ in range(n-1):
            x = self.performTransitionModel()
        return x.value[0][0], x.value[1][0]

    def getX(self):
        """ Get the current estimation of the x position. """
        return self.x.value[0][0]

    def getY(self):
        """ Get the current estimation of the y position. """
        return self.x.value[1][0]

    def getDirection(self):
        """
        Get the current estimation of the heading direction in radians.
        """
        return self.x.value[2][0]

    def getTurningAngle(self):
        """ Get the current estimation of the turning angle. """
        return self.x.value[3][0]

    def getDistance(self):
        """ Get the current estimation of the distance traveled. """
        return self.x.value[4][0]

# This is the function you have to write. Note that measurement is a 
# single (x, y) point. This function will have to be called multiple
# times before you have enough information to accurately predict the
# next position. The OTHER variable that your function returns will be 
# passed back to your function the next time it is called. You can use
# this to keep track of important information over time.
def estimate_next_pos(measurement, OTHER=None):
    if not OTHER:
        OTHER = KalmanFilter()

    z = matrix([[measurement[0]], [measurement[1]]])
    
    OTHER.updateMeasurement(z)
    x, y = OTHER.predict()

    return (x, y), OTHER


# A helper function you may find useful.
def distance_between(point1, point2):
    """Computes distance between point1 and point2. Points are (x, y) pairs."""
    x1, y1 = point1
    x2, y2 = point2
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# This is here to give you a sense for how we will be running and grading
# your code. Note that the OTHER variable allows you to store any 
# information that you want. 
def demo_grading(estimate_next_pos_fcn, target_bot, OTHER = None):
    localized = False
    distance_tolerance = 0.01 * target_bot.distance
    ctr = 0
    # if you haven't localized the target bot, make a guess about the next
    # position, then we move the bot and compare your guess to the true
    # next position. When you are close enough, we stop checking.
    while not localized and ctr <= 1000:
        ctr += 1
        measurement = target_bot.sense()
        position_guess, OTHER = estimate_next_pos_fcn(measurement, OTHER)
        target_bot.move_in_circle()
        true_position = (target_bot.x, target_bot.y)
        error = distance_between(position_guess, true_position)
        if error <= distance_tolerance:
            print("You got it right! It took you ", ctr, " steps to localize.")
            localized = True
        if ctr == 1000:
            print("Sorry, it took you too many steps to localize the target.")
    return localized

# This is a demo for what a strategy could look like. This one isn't very good.
def naive_next_pos(measurement, OTHER = None):
    """This strategy records the first reported position of the target and
    assumes that eventually the target bot will eventually return to that 
    position, so it always guesses that the first position will be the next."""
    if not OTHER: # this is the first measurement
        OTHER = measurement
    xy_estimate = OTHER 
    return xy_estimate, OTHER

# This is how we create a target bot. Check the robot.py file to understand
# How the robot class behaves.
test_target = robot(2.1, 4.3, 0.5, 2*pi / 34.0, 1.5)
measurement_noise = 0.05 * test_target.distance
test_target.set_noise(0.0, 0.0, measurement_noise)

demo_grading(estimate_next_pos, test_target)