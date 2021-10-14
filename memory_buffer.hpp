#ifndef MEMORY_BUFFER_HPP
#define MEMORY_BUFFER_HPP


#include <iostream>
#include <eigen3/Eigen/Eigen>



class memory_buffer{

    private:

        std::vector<Eigen::RowVectorXf*> vObservation;
        std::vector<Eigen::RowVectorXf*> vObservation1;
        std::vector<int> vAction;
        std::vector<bool> vDone;


    public:
    
        memory_buffer();

        ~memory_buffer();

        void add(Eigen::RowVectorXf& obs, Eigen::RowVectorXf& obs1, int act, bool bdone);

        int size();

        
        Eigen::RowVectorXf* sample_observation(int idx);
        Eigen::RowVectorXf* sample_observation1(int idx);
        int sample_action(int idx);
        bool sample_done(int idx);

        void display_memory(int idx);

};

#endif