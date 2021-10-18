#ifndef ENV_MOUNTAIN_CAR_CONT_HPP
#define ENV_MOUNTAIN_CAR_CONT_HPP

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
    Type: Box(2)
    Num     Action                    Min                     Max
    0       the power coef            -1.0                    1.0
    Note: actual driving force is calculated by multipling the power coef by power (0.0015)
Reward:
    Reward of 100 is awarded if the agent reached the flag (position = 0.45) on top of the mountain.
    Reward is decrease based on amount of energy consumed each step.
Starting State:
    The position of the car is assigned a uniform random value in [-0.6 , -0.4].
    The starting velocity of the car is always assigned to 0.
Episode Termination:
     The car position is more than 0.45
    Episode length is greater than 200
*/




class env_mountain_car_cont{

    private:

        bool bDebug = false;
        std::string envId = "MountainCarCont";
        std::string actType = "continuous";

        float action_min = -1.0;
        float action_max = 1.0;
        float position_min = -1.2;
        float position_max = 0.6;
        float speed_max = 0.07;
        float position_goal = 0.45;

        float power = 0.0015;
        float velocity_goal = 0.0;


        float x = 0.0; // position
        float v = 0.0; // speed
        Eigen::RowVectorXf state = Eigen::RowVectorXf(2);

        // Render 
        cv::Mat img = cv::Mat(600,800,CV_8UC3, cv::Scalar(255,255,255));

    public:

        env_mountain_car_cont(bool bdbg = false);
        ~env_mountain_car_cont();

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