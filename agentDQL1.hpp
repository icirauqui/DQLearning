#ifndef AGENTDQL_HPP
#define AGENTDQL_HPP



#include <iostream>

//#include "environments/env_warehouse.cpp"
#include "environments/env_cart_pole.cpp"
#include "NN.cpp"
#include "memory_buffer.cpp"

#include <eigen3/Eigen/Eigen>
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;


class agentDQL{

    private:

        int row, row1, col, col1;

        std::vector<int> topology;

        float learning_rate = 0.9;
        float discount_factor = 0.9;
        float epsilon = 0.9;
        float epsilon1 = 0.0;

        bool bDebug;

        // Environment
        env_cart_pole* pEnv;
        // Main NN
        NN *pNN1;
        // Target NN
        NN *pNN2;
        // Memory
        memory_buffer *pMemory;

    public:


        agentDQL(std::vector<int> topology1, float learningRate, float discount_factor, float epsilon);

        ~agentDQL();

        void debug_mode(bool bDebug = false);

        bool is_terminal_state();
        void get_starting_location();
        void get_next_action(bool exploration = true);
        void get_next_location();
        
        void train(int num_episodes, int max_steps, int target_upd, int exp_upd);
        void test(int max_steps, int num_episodes = 1, bool verbose = false);

        int select_epsilon_greedy_action(RowVector& obs, bool bTrain = true);

        void epsilon_decay();
   
        void experience_replay(int update_size);

        void backup_epsilon();
        void restore_epsilon();

};


#endif