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
        int row, col;
        
        // Create a 2D array to hold the rewards for each state
        // The array is 11x11, matching the shape of the environment
        std::vector<std::vector<int> > rewards;

    
        bool bDebug = false;

        std::vector<int> env_dims;

        // Number of steps taken in current path
        int path_steps;


    public:
        env_warehouse();
        ~env_warehouse();

        void debug_mode(bool bDebug);

        RowVector* reset();

        std::vector<int> get_env_dims();

        int get_reward(int row, int col);

        void render();

        void step(int action, RowVector &observation, float &reward, bool &done);

        bool is_terminal_state();

};


#endif