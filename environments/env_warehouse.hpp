#ifndef ENV_WAREHOUSE_HPP
#define ENV_WAREHOUSE_HPP


#include <iostream>
#include <vector>
#include <fstream>

#include <eigen3/Eigen/Eigen>

typedef float Scalar;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;


class env_warehouse{
    private:
        bool bDebug = false;
        std::string envId = "Warehouse";
        
        // Create a 2D array to hold the rewards for each state
        // The array is 11x11, matching the shape of the environment
        std::vector<std::vector<int> > rewards;

        std::vector<int> env_dims;

        // State
        int row, col;
        Eigen::RowVectorXf state = Eigen::RowVectorXf(2);

        // Discrete
        std::string actType = "discrete";
        std::vector<int> action_space = {0,1, 2, 3};

        // Number of steps taken in current path
        int path_steps;


    public:
        env_warehouse();
        ~env_warehouse();

        void debug_mode(bool dbg);
        std::string get_env_id();
        std::string get_env_actType();
        std::vector<int> get_env_dims();

        RowVector* reset();

        int get_reward(int row, int col);

        void render();

        void step(int action, RowVector &observation, float &reward, bool &done);

        bool is_terminal_state();

};


#endif