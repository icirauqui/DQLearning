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



void environment::debug_mode(bool bDebug){
    this->bDebug = bDebug;
}



RowVector* environment::reset(){
    // Determines a random not terminal starting location    
    int rowt;
    int colt;
            
    do {
        rowt = rand() % 11;
        colt = rand() % 11;
    }
    while (get_reward(rowt,colt) != -1);

    row = rowt;
    col = colt;

    RowVector* obs = new RowVector(8);
    if (is_terminal_state()){
        for (int i=0; i<obs->size(); i++){
            obs->coeffRef(i) = -100;
        }
    }
    else{
        obs->coeffRef(0) = rewards[row-1][col-1];
        obs->coeffRef(1) = rewards[row-1][col];
        obs->coeffRef(2) = rewards[row-1][col+1];
        obs->coeffRef(3) = rewards[row][col-1];
        obs->coeffRef(4) = rewards[row][col+1];
        obs->coeffRef(5) = rewards[row+1][col-1];
        obs->coeffRef(6) = rewards[row+1][col];
        obs->coeffRef(7) = rewards[row+1][col+1];
    }

    path_steps = 0;
    
    //std::vector<int> start_location = std::vector<int>{row,col};
    return obs;
}



std::vector<int> environment::get_env_dims(){
    return env_dims;
}


int environment::get_reward(int row, int col){
    return rewards[row][col];
}



void environment::render(){
    std::cout << "Location = (" << row << " " << col << ") " << path_steps + 1 << std::endl;

}



void environment::step(RowVector &observation, float &reward, bool &done, int action){
    // Get new location based on last action
    int row1 = row;
    int col1 = col;
    if (action==0 and row>0)
        row1 -= 1;
    else if (action==1 and col<(env_dims[1]-1))
        col1 += 1;
    else if (action==2 and row<(env_dims[0]-1))
        row1 += 1;
    else if (action==3 and col>0)
        col1 -= 1;
    row = row1;
    col = col1;

    path_steps += 1;

    //observation.coeffRef(0) = row;
    //observation.coeffRef(1) = col;

    if (is_terminal_state()){
        for (int i=0; i<observation.size(); i++){
            observation.coeffRef(i) = -100;
        }
    }
    else{
        observation.coeffRef(0) = rewards[row-1][col-1];
        observation.coeffRef(1) = rewards[row-1][col];
        observation.coeffRef(2) = rewards[row-1][col+1];
        observation.coeffRef(3) = rewards[row][col-1];
        observation.coeffRef(4) = rewards[row][col+1];
        observation.coeffRef(5) = rewards[row+1][col-1];
        observation.coeffRef(6) = rewards[row+1][col];
        observation.coeffRef(7) = rewards[row+1][col+1];
    }




    reward = rewards[row][col];
    done = is_terminal_state();
}



bool environment::is_terminal_state(){
    // Determines if specified location is a terminal state
    if (rewards[row][col]==-1)
        return false;
    else
        return true;
}

