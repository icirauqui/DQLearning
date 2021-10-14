#include "env_cart_pole.hpp"


env_cart_pole::env_cart_pole(std::string kinematics){
    this->kinematics_integrator = kinematics;
}


env_cart_pole::~env_cart_pole(){}


void env_cart_pole::debug_mode(bool dbg){
    this->bDebug = dbg;
}



void env_cart_pole::step(int action, Eigen::RowVectorXf &zstate, float &zreward, bool &zdone){
    float force;
    
    if (action==1) 
        force = force_mag;
    else
        force = -force_mag;
    

    float costheta = cos(theta);
    float sintheta = sin(theta);

    float temp = (force + mass_pole_length * pow(thetadot,2) * sintheta) / mass_total;
    float thetaacc = (gravity * sintheta - costheta * temp) / (length * (4.0 / 3.0 - mass_pole * pow(costheta,2) / mass_total));
    float xacc = temp - mass_pole_length * thetaacc * costheta / mass_total;

    if (kinematics_integrator == "euler") {
        x = x + tau * xdot;
        xdot = xdot + tau * xacc;
        theta = theta + tau * thetadot;
        thetadot = thetadot + tau * thetaacc;
    }
    else{   // semi-implicit euler
        xdot = xdot + tau * xacc;
        x = x + tau * xdot;
        thetadot = thetadot + tau * thetaacc;
        theta = theta + tau * thetadot;
    }

    state.coeffRef(0) = x;
    state.coeffRef(1) = xdot;
    state.coeffRef(2) = theta;
    state.coeffRef(3) = thetadot;

    bool done = false;
    if (x < -x_threshold || x > x_threshold || theta < -theta_threshold_radians || theta > theta_threshold_radians)
        done = true;
    else 
        done = false;

    float reward = 0.0;
    if (!done)
        reward = 1.0;
    else if (steps_beyond_done == -1){
        steps_beyond_done = 0;
        reward = 1.0;
    }
    else{
        if (steps_beyond_done == 0){
            std::cout << "You're calling step() even though this environment has already returned done = True." << std::endl
                      << "You should always call reset() once you receive done = True, any further steps are undefined behavior." << std::endl;
        }
        steps_beyond_done += 1;
        reward = 0.0;
    }


    zstate = state;
    zreward = reward;
    zdone = done;
}

Eigen::RowVectorXf* env_cart_pole::reset(){

    float LO = -0.05;
    float HI = +0.05;
    x = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
    xdot = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
    theta = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
    thetadot = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));

    /*
    std::vector<float> initstate = std::vector<float> (4,0);    
    std::ifstream fp0("init_state.csv");
    for (unsigned int i=0; i<initstate.size(); i++)
        fp0 >> initstate[i];

    x = initstate[0];
    xdot = initstate[1];
    theta = initstate[2];
    thetadot = initstate[3];
    */
    

    state.coeffRef(0) = x;
    state.coeffRef(1) = xdot;
    state.coeffRef(2) = theta;
    state.coeffRef(3) = thetadot;

    
    steps_beyond_done = -1;
    return &state;
}

void env_cart_pole::render(int framerate){
    float x1 = 400*x;
    float th1 = 1*theta;

    img = cv::Mat(600,800,CV_8UC3, cv::Scalar(255,255,255));
    for (int i=0; i<img.cols; i++){
        img.at<cv::Vec3b>(cv::Point(i,0.8*img.rows)) = cv::Vec3b(0.0,0.0,0.0);
    }

    int height = 40;
    int width = 80;
    int pole_length = 300*length;

    cv::Point pt1 = cv::Point(0.5*img.cols+x1, 0.8*img.rows-height/2);
    cv::Point pt2 = cv::Point(0.5*img.cols+x1+sin(th1)*pole_length,0.8*img.rows-height/2-cos(th1)*pole_length);
    
    cv::rectangle(img,cv::Point(0.5*img.cols+x1-width/2,0.8*img.rows-height/2),cv::Point(0.5*img.cols+x1+width/2,0.8*img.rows+height/2),cv::Scalar(0,0,0),cv::FILLED,cv::LINE_8);
    cv::line(img,pt1,pt2,cv::Scalar(50,180,230),12,cv::LINE_8);
    cv::circle(img,pt1,6,cv::Scalar(120,120,120),cv::FILLED,cv::LINE_8);

    cv::namedWindow("Render",cv::WINDOW_AUTOSIZE);
    cv::imshow("Render",img);
    cv::waitKey(framerate);
}
