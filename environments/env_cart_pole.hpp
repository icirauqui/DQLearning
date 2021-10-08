#ifndef ENV_CART_POLE_HPP
#define ENV_CART_POLE_HPP

#include <iostream>
#include <vector>
#include <math.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <eigen3/Eigen/Eigen>

#define PI 3.14159265358979323846


/*
Description:
    A pole is attached by an un-actuated joint to a cart, which moves along
    a frictionless track. The pendulum starts upright, and the goal is to
    prevent it from falling over by increasing and reducing the cart's
    velocity.
Source:
    This environment corresponds to the version of the cart-pole problem
    described by Barto, Sutton, and Anderson
Observation:
    Type: Box(4)
    Num     Observation               Min                     Max
    0       Cart Position             -4.8                    4.8
    1       Cart Velocity             -Inf                    Inf
    2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
    3       Pole Angular Velocity     -Inf                    Inf
Actions:
    Type: Discrete(2)
    Num   Action
    0     Push cart to the left
    1     Push cart to the right
    Note: The amount the velocity that is reduced or increased is not
    fixed; it depends on the angle the pole is pointing. This is because
    the center of gravity of the pole increases the amount of energy needed
    to move the cart underneath it
Reward:
    Reward is 1 for every step taken, including the termination step
Starting State:
    All observations are assigned a uniform random value in [-0.05..0.05]
Episode Termination:
    Pole Angle is more than 12 degrees.
    Cart Position is more than 2.4 (center of the cart reaches the edge of
    the display).
    Episode length is greater than 200.
    Solved Requirements:
    Considered solved when the average return is greater than or equal to
    195.0 over 100 consecutive trials.
*/




class env_cart_pole{


    private:

        float gravity = 9.8;
        float mass_cart = 1.0;
        float mass_pole = 0.1;
        float mass_total = mass_cart + mass_pole;
        float length = 0.5; // half the pole's length
        float mass_pole_length = mass_pole * length;
        float force_mag = 10.0;
        float tau = 0.02;   // seconds between state updates

        // angle at which to fail the episode
        float theta_threshold_radians = 12 * 2 * PI / 360;
        float x_threshold = 2.4;

        std::vector<float> high = {2*x_threshold, static_cast <float> (RAND_MAX), theta_threshold_radians, static_cast <float> (RAND_MAX)};
        std::vector<float> low = {-2*x_threshold, -1*static_cast <float> (RAND_MAX), -theta_threshold_radians, -1*static_cast <float> (RAND_MAX)};

        float x = 0.0; // position
        float xdot = 0.0; // speed
        float theta = 0.0; // angle
        float thetadot = 0.0; // angular velocity
        Eigen::RowVectorXf state = Eigen::RowVectorXf(4);

        std::vector<int> action_space = {0,1};

        int steps_beyond_done = -1;

        std::vector<std::vector<float> > observation_space = {high,low};

        std::string kinematics_integrator = "euler";


        // Render 
        cv::Mat img = cv::Mat(600,800,CV_8UC3, cv::Scalar(255,255,255));

        bool bDebug = false;

    public:

        env_cart_pole(std::string kinematics = "euler");
        ~env_cart_pole();

        void debug_mode(bool dbg);

        void step(int action, Eigen::RowVectorXf &zstate, float &zreward, bool &zdone);

        //std::vector<float> reset();
        Eigen::RowVectorXf* reset();

        void render(int framerate = 33);








};



#endif