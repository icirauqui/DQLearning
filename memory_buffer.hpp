#ifndef MEMORY_BUFFER_HPP
#define MEMORY_BUFFER_HPP


#include <iostream>
#include <eigen3/Eigen/Eigen>



class memory_buffer{

    private:

        std::vector<Eigen::RowVectorXf*> vObservation;
        std::vector<Eigen::RowVectorXf*> vObservation1;
        std::vector<float> vAction;
        std::vector<bool> vDone;
        std::vector<float> vReward;


    public:
    
        memory_buffer();

        ~memory_buffer();

        void add(Eigen::RowVectorXf& obs, Eigen::RowVectorXf& obs1, float act, bool bdone, float reward);

        int size();

        
        Eigen::RowVectorXf* sample_observation(int idx);
        Eigen::RowVectorXf* sample_observation1(int idx);
        float sample_action(int idx);
        bool sample_done(int idx);
        float sample_reward(int idx);

        void display_memory(int idx);

        void forget_memory(int idx);
        void clear_memory();

};

#endif