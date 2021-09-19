#ifndef AGENTDQL_HPP
#define AGENTDQL_HPP



#include <iostream>

#include "environment.cpp"
#include "DNN.cpp"
#include "memory_buffer.cpp"



class agentDQL{

    private:

        int row, row1, col, col1;

        std::vector<int> topology;

        float learning_rate = 0.9;
        float discount_factor = 0.9;
        float epsilon = 0.9;

        bool bDebug;

        // Environment
        environment* pEnv;
        // Main NN
        DNN *pDNN1;
        // Target NN
        DNN *pDNN2;
        // Memory
        memory_buffer *pMemory;

    public:


        agentDQL(std::vector<int> env_dims, std::vector<int> topology1, float learningRate, float discount_factor, float epsilon);

        ~agentDQL();

        void debug_mode(bool bDebug = false);

        bool is_terminal_state();
        void get_starting_location();
        void get_next_action(bool exploration = true);
        void get_next_location();
        
        void train(int num_episodes, int max_steps, int target_upd, int exp_upd);

        int select_epsilon_greedy_action(RowVector& obs);

        void epsilon_decay();
   
        void experience_replay(int update_size);

};


#endif