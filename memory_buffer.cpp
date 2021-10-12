#include "memory_buffer.hpp"

memory_buffer::memory_buffer(){};


memory_buffer::~memory_buffer(){};


void memory_buffer::add(Eigen::RowVectorXf& obs, Eigen::RowVectorXf& obs1, int act, bool bdone){

    // Duplicate the observations with new pointers to the heap, otherwise the original pointer info will be updated, 
    // thus we'd only save the last data.
    Eigen::RowVectorXf* obst = new Eigen::RowVectorXf(obs.size());
    *obst = obs;
    vObservation.push_back(obst);
    //vObservation.push_back(&obs);

    Eigen::RowVectorXf* obst1 = new Eigen::RowVectorXf(obs1.size());
    *obst1 = obs1;
    vObservation1.push_back(obst1);
    //vObservation1.push_back(&obs1);

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


void memory_buffer::display_memory(int idx){
    std::cout << "Memory[" << idx << "] = ";
    for (unsigned int i=0; i<vObservation[idx]->size(); i++)
        std::cout << vObservation[idx]->coeffRef(i) << " ";
    std::cout << "| ";
    for (unsigned int i=0; i<vObservation1[idx]->size(); i++){
        std::cout << vObservation1[idx]->coeffRef(i) << " ";
    }
    std::cout << "| " << vAction[idx] << " | " << vDone[idx] << std::endl;
}