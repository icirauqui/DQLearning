#include "environment.hpp"

environment::environment(std::vector<int> env_dims){

    this->env_dims = env_dims;

    // Fill rewards from csv file
    rewards = std::vector<std::vector<int> > (env_dims[0], std::vector<int>(env_dims[1],0));
    std::ifstream fp("board.csv");
    for (unsigned int i=0; i<env_dims[0]; i++)
        for (unsigned int j=0; j<env_dims[1]; j++)
            fp >> rewards[i][j];


}


environment::~environment(){}






std::vector<int> environment::get_env_dims(){
    return env_dims;
}