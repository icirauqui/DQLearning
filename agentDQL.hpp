#ifndef AGENTDQL_HPP
#define AGENTDQL_HPP



#include <iostream>
#include <fstream>

//#include "environments/env_warehouse.cpp"
//#include "environments/env_cart_pole.cpp"
#include "environments/environments.hpp"
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
        std::vector<std::string> act_funcs;

        float learning_rate = 0.9;
        float gamma = 0.9;
        float epsilon = 1.0;
        float epsilon1 = 0.0;

        float eps_min = 0.01;
        float eps_decay = 0.997;

        bool bDebug;

        // Environment
        //env_warehouse* pEnv = new env_warehouse();
        env_cart_pole* pEnv = new env_cart_pole();
        //env_mountain_car* pEnv = new env_mountain_car();
        //env_mountain_car_cont* pEnv = new env_mountain_car_cont();
        int envType = 1;    // 1 = Discrete, 2 = Continuous

        // Main NN
        NN *pNN1;
        // Target NN
        NN *pNN2;
        // Memory
        memory_buffer *pMemory;

    public:


        agentDQL(std::vector<int> topology1, std::vector<std::string> act_funcs1, float learningRate, float gamma, float epsilon);

        ~agentDQL();

        void debug_mode(bool bDebug = false);

        bool is_terminal_state();
        void get_starting_location();
        void get_next_action(bool exploration = true);
        void get_next_location();
        
        void train(int num_episodes, int max_steps, int target_upd, int exp_upd);
        void test(int max_steps, int num_episodes = 1, bool verbose = false);

        float select_epsilon_greedy_action(RowVector& obs, bool bTrain = true);

        void epsilon_decay();
   
        void experience_replay(int update_size);
        void shuffle(int *arr, size_t n);

        void backup_epsilon();
        void restore_epsilon();

        void save_model();
        void load_model();

};



#endif