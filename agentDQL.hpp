#include <iostream>

#include "environment.hpp"
#include "DNN.cpp"



class agentDQL{

    private:

        int row, row1, col, col1;
        int action, action1;

        std::vector<uint> topology;

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
        std::vector<std::vector<int> > memory_observation;
        std::vector<std::vector<int> > memory_observation1;
        std::vector<int> memory_action;
        std::vector<bool> memory_done;






    public:


        agentDQL(environment* pEnv, std::vector<uint> topology, float learningRate, float discount_factor, float epsilon);

        ~agentDQL();

        void debug_mode(bool bDebug = false);


        bool is_terminal_state();
        void get_starting_location();
        void get_next_action(bool exploration = true);
        void get_next_location();
        
        void train(int num_episodes, int max_steps, int target_upd, int exp_upd);

        int select_action(std::vector<int> obs);

        void epsilon_decay();
   
        void experience(int update_size);

};