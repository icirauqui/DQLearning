#include "memory_buffer.hpp"

memory_buffer::memory_buffer(){};


memory_buffer::~memory_buffer(){};


void memory_buffer::add(Eigen::RowVectorXf& obs, Eigen::RowVectorXf& obs1, int act, bool bdone){
    vObservation.push_back(&obs);
    vObservation1.push_back(&obs1);
    vAction.push_back(act);
    vDone.push_back(bdone);
}


int memory_buffer::size(){
    return vObservation.size();
}

        
Eigen::RowVectorXf* memory_buffer::sample_observation(int idx){
    return vObservation[idx];
}

        
Eigen::RowVectorXf* memory_buffer::sample_observation1(int idx){
    return vObservation1[idx];
}


int memory_buffer::sample_action(int idx){
    return vAction[idx];
}


bool memory_buffer::sample_done(int idx){
    return vDone[idx];
}
