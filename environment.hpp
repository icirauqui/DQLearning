#include <iostream>
#include <vector>
#include <fstream>

#include <eigen3/Eigen/Eigen>

typedef float Scalar;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;


class environment{
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
        environment(std::vector<int> env_dims);
        ~environment();

        void debug_mode(bool bDebug);

        RowVector* reset();

        std::vector<int> get_env_dims();

        int get_reward(int row, int col);

        void render();

        void step(std::vector<int> &observation, float &reward, bool &done, int action);

        bool is_terminal_state();

};
