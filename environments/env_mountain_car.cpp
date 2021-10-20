#include "env_mountain_car.hpp"

env_mountain_car::env_mountain_car(bool bdbg){
    this->bDebug = bdbg;
}

env_mountain_car::~env_mountain_car(){}



void env_mountain_car::debug_mode(bool dbg){
    this->bDebug = dbg;
}

std::string env_mountain_car::get_env_id(){
    return envId;
}

std::string env_mountain_car::get_env_actType(){
    return actType;
}

std::vector<int> env_mountain_car::get_env_dims(){
    std::vector<int> env_dims;
    env_dims.push_back(state.size());
    env_dims.push_back(action_space.size());
    return env_dims;
}

void env_mountain_car::step(float action, Eigen::RowVectorXf &zstate, float &zreward, bool &zdone){
    x = state.coeffRef(0);
    v = state.coeffRef(1);

    v += (action-1) * force + cos(3*x)*(-gravity);
    if (v > speed_max)
        v = speed_max;
    if (v < -speed_max)
        v = -speed_max;

    x += v;
    if (x > position_max)
        x = position_max;
    if (x < position_min)
        x = position_min;
    if (x == position_min && v < 0)
        v = 0.0;

    bool done = false;
    if (x >= position_goal && v >= velocity_goal)
        done = true;
    else 
        done = false;

    float reward = -1.0;
    reward = x - (-0.6);

    state.coeffRef(0) = x;
    state.coeffRef(1) = v;
    zstate = state;
    zreward = reward;
    zdone = done;
}


Eigen::RowVectorXf* env_mountain_car::reset(){
    float LO = -0.6;
    float HI = -0.4;
    state.coeffRef(0) = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
    state.coeffRef(1) = 0.0;

    return &state;
}


std::vector<float> env_mountain_car::_height(std::vector<float> xs){
    std::vector<float> ys;
    for (unsigned int i=0; i<xs.size(); i++){
        float xs1 = sin(3 * xs[i]) * 0.45 + 0.55;
        ys.push_back(xs1);
    }
    return ys;
}


float env_mountain_car::_height(float xs){
    float ys = sin(3 * xs) * 0.45 + 0.55;
    return ys;
}


void env_mountain_car::render(int framerate){

    float screen_width = 600;
    float screen_height = 400;

    float world_width = position_max - position_min;
    float scale = screen_width / world_width;
    float carwidth = 40;
    float carheight = 20;
    float wheelrad = carheight/2.5;



    img = cv::Mat(screen_height,screen_width,CV_8UC3, cv::Scalar(255,255,255));

    std::vector<float> xpoly, ypoly;
    float pt1 = position_min;
    xpoly.push_back(pt1);
    for (unsigned int i=0; i<100; i++){
        pt1 += (position_max - position_min)/100;
        xpoly.push_back(pt1);
    }
    ypoly = _height(xpoly);

    std::vector<cv::Point> contour;
    for (unsigned int i=0; i<xpoly.size(); i++)
        contour.push_back(cv::Point(scale*(xpoly[i]-position_min),screen_height - scale*ypoly[i]));

    const cv::Point *pts = (const cv::Point*) cv::Mat(contour).data;
    int npts = cv::Mat(contour).rows;

    polylines(img, &pts, &npts, 1, false, cv::Scalar(0, 0, 0), 3);

    float clearance = 10;

    float l = -carwidth / 2;
    float r = carwidth / 2;
    float t = carheight;
    float b = 0.0;


    float x = state.coeffRef(0);
    float y = _height(x);
    float xpos = scale*(x-position_min);
    float ypos = screen_height - scale*y;
    //std::cout << x << " " << vxpos[0] << " " << xpos << std::endl;
    //std::cout << vypos[0] << " " << ypos << std::endl;

    cv::Point rook_points[1][4];
    rook_points[0][0]  = cv::Point(xpos+l, ypos+b);
    rook_points[0][1]  = cv::Point(xpos+l, ypos+t);
    rook_points[0][2]  = cv::Point(xpos+r, ypos+t);
    rook_points[0][3]  = cv::Point(xpos+r, ypos+b);

    const cv::Point* ppt[1] = { rook_points[0] };
    int npt[] = { 4 };
    fillPoly(img, ppt, npt, 1, cv::Scalar(0,0,0), cv::LINE_8 );







    cv::namedWindow("Render",cv::WINDOW_AUTOSIZE);
    cv::imshow("Render",img);
    cv::waitKey(framerate);
}