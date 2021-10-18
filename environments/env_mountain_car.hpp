#ifndef ENV_MOUNTAIN_CAR_HPP
#define ENV_MOUNTAIN_CAR_HPP

#include <iostream>
#include <vector>
#include <math.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <eigen3/Eigen/Eigen>

#define PI 3.14159265358979323846


/*
Description:
    The agent (a car) starts at the bottom of a valley. For any given state 
    the agent may choose to accelerate to the left, right, or not accelerate.
Source:
    Adaptation of the MountainCar Environment from the "FAReinforcement" library
    of Jose Antonio Martin H. (version 1.0), adapted by  'Tom Schaul, tom@idsia.ch'
    and then modified by Arnaud de Broissia
Observation:
    Type: Box(4)
    Num     Observation               Min                     Max
    0       Car Position              -1.2                    0.6
    1       Car Velocity              -0.07                   0.07
Actions:
        Type: Discrete(3)
        Num    Action
        0      Accelerate to the Left
        1      Don't accelerate
        2      Accelerate to the Right
        Note: This does not affect the amount of velocity affected by the
        gravitational pull acting on the car.
Reward:
         Reward of 0 is awarded if the agent reached the flag (position = 0.5)
         on top of the mountain.
         Reward of -1 is awarded if the position of the agent is less than 0.5.
Starting State:
         The position of the car is assigned a uniform random value in
         [-0.6 , -0.4].
         The starting velocity of the car is always assigned to 0.
Episode Termination:
         The car position is more than 0.5
         Episode length is greater than 200
*/




class env_mountain_car{

    private:

        bool bDebug = false;
        std::string envId = "MountainCar";
        std::string actType = "discrete";

        float position_min = -1.2;
        float position_max = 0.6;
        float speed_max = 0.07;
        float position_goal = 0.45;
        float velocity_goal = 0.0;

        float force = 0.001;
        float gravity = 0.0025;


        float x = 0.0; // position
        float v = 0.0; // speed
        Eigen::RowVectorXf state = Eigen::RowVectorXf(2);

        // Render 
        cv::Mat img = cv::Mat(600,800,CV_8UC3, cv::Scalar(255,255,255));

    public:

        env_mountain_car(bool bdbg = false);
        ~env_mountain_car();

        void debug_mode(bool dbg);
        std::string get_env_id();
        std::string get_env_actType();

        void step(float action, Eigen::RowVectorXf &zstate, float &zreward, bool &zdone);

        Eigen::RowVectorXf* reset();

        std::vector<float> _height(std::vector<float> xs);
        float _height(float xs);
        void render(int framerate = 33);


};



#endif