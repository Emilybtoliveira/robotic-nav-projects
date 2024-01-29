# ----------
# Part Five
#
# This time, the sensor measurements from the runaway Traxbot will be VERY 
# noisy (about twice the target's stepsize). You will use this noisy stream
# of measurements to localize and catch the target.
#
# ----------
# YOUR JOB
#
# Complete the next_move function, similar to how you did last time. 
#
# ----------
# GRADING
# 
# Same as part 3 and 4. Again, try to catch the target in as few steps as possible.

from robot import *
from math import *
from matrix import *


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

def next_move(hunter_position, hunter_heading, target_measurement, max_distance, OTHER = None):
    # This function will be called after each time the target moves. 

    # The OTHER variable is a place for you to store any historical information about
    # the progress of the hunt (or maybe some localization information). Your return format
    # must be as follows in order to be graded properly.
    if not OTHER:
        OTHER = {}
        OTHER['EKF'] = KalmanFilter()
        OTHER['steps'] = 0

    z = matrix([[target_measurement[0]],[target_measurement[1]]])
    
    OTHER['EKF'].updateMeasurement(z)
    target_position = OTHER['EKF'].predict()
    
    if OTHER['steps'] < 1:
        # keep predicting the next step until our movements cover the distance
        nSteps = 1
        while distance_between(hunter_position, target_position) > max_distance * nSteps:
            target_position = OTHER['EKF'].predictAfterN(1)
            nSteps += 1
        OTHER['steps'] = nSteps
    else:
        # otherwise, refine the predicted capture position
        OTHER['steps'] -= 1
        target_position = OTHER['EKF'].predictAfterN(OTHER['steps'])
 
    diff = get_heading(hunter_position, target_position) - hunter_heading
    distance = min(max_distance, distance_between(hunter_position, target_position))
    
    return diff, distance, OTHER

def distance_between(point1, point2):
    """Computes distance between point1 and point2. Points are (x, y) pairs."""
    x1, y1 = point1
    x2, y2 = point2
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def demo_grading(hunter_bot, target_bot, next_move_fcn, OTHER = None):
    """Returns True if your next_move_fcn successfully guides the hunter_bot
    to the target_bot. This function is here to help you understand how we 
    will grade your submission."""
    max_distance = 0.97 * target_bot.distance # 0.97 is an example. It will change.
    separation_tolerance = 0.02 * target_bot.distance # hunter must be within 0.02 step size to catch target
    caught = False
    ctr = 0

    # We will use your next_move_fcn until we catch the target or time expires.
    while not caught and ctr < 1000:

        # Check to see if the hunter has caught the target.
        hunter_position = (hunter_bot.x, hunter_bot.y)
        target_position = (target_bot.x, target_bot.y)
        separation = distance_between(hunter_position, target_position)
        if separation < separation_tolerance:
            print("You got it right! It took you ", ctr, " steps to catch the target.")
            caught = True

        # The target broadcasts its noisy measurement
        target_measurement = target_bot.sense()

        # This is where YOUR function will be called.
        turning, distance, OTHER = next_move_fcn(hunter_position, hunter_bot.heading, target_measurement, max_distance, OTHER)
        
        # Don't try to move faster than allowed!
        if distance > max_distance:
            distance = max_distance

        # We move the hunter according to your instructions
        hunter_bot.move(turning, distance)

        # The target continues its (nearly) circular motion.
        target_bot.move_in_circle()

        ctr += 1            
        if ctr >= 1000:
            print("It took too many steps to catch the target.")
    return caught



def angle_trunc(a):
    """This maps all angles to a domain of [-pi, pi]"""
    while a < 0.0:
        a += pi * 2
    return ((a + pi) % (pi * 2)) - pi

def get_heading(hunter_position, target_position):
    """Returns the angle, in radians, between the target and hunter positions"""
    hunter_x, hunter_y = hunter_position
    target_x, target_y = target_position
    heading = atan2(target_y - hunter_y, target_x - hunter_x)
    heading = angle_trunc(heading)
    return heading

def naive_next_move(hunter_position, hunter_heading, target_measurement, max_distance, OTHER):
    """This strategy always tries to steer the hunter directly towards where the target last
    said it was and then moves forwards at full speed. This strategy also keeps track of all 
    the target measurements, hunter positions, and hunter headings over time, but it doesn't 
    do anything with that information."""
    if not OTHER: # first time calling this function, set up my OTHER variables.
        measurements = [target_measurement]
        hunter_positions = [hunter_position]
        hunter_headings = [hunter_heading]
        OTHER = (measurements, hunter_positions, hunter_headings) # now I can keep track of history
    else: # not the first time, update my history
        OTHER[0].append(target_measurement)
        OTHER[1].append(hunter_position)
        OTHER[2].append(hunter_heading)
        measurements, hunter_positions, hunter_headings = OTHER # now I can always refer to these variables
    
    heading_to_target = get_heading(hunter_position, target_measurement)
    heading_difference = heading_to_target - hunter_heading
    turning =  heading_difference # turn towards the target
    distance = max_distance # full speed ahead!
    return turning, distance, OTHER

target = robot(0.0, 10.0, 0.0, 2*pi / 30, 1.5)
measurement_noise = 2.0*target.distance # VERY NOISY!!
target.set_noise(0.0, 0.0, measurement_noise)

hunter = robot(-10.0, -10.0, 0.0)

print(demo_grading(hunter, target, next_move))