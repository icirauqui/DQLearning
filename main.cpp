#include <iostream>

#include "environments/env_mountain_car_cont.cpp"

int main(){

    env_mountain_car_cont* pEnv = new env_mountain_car_cont();

    pEnv->reset();
    pEnv->render(100);

    Eigen::RowVectorXf state(2);
    float reward = 0.0;
    bool done = false;

    for (unsigned int i=0; i<1000; i++){
        pEnv->step(1.0,state,reward,done);
        pEnv->render(100);
        std::cout << state << std::endl;
    }

    return 0;
}
























