#include "env_warehouse.hpp"

env_warehouse::env_warehouse(){

    this->env_dims = {11,11};

    // Fill rewards
    rewards.clear();
    std::vector<int> vv01 = {-100, -100, -100, -100, -100,  100, -100, -100, -100, -100, -100};     rewards.push_back(vv01);
    std::vector<int> vv02 = {-100, -1  , -1  , -1  , -1  , -1  , -1  , -1  , -1  , -1  , -100};     rewards.push_back(vv02);
    std::vector<int> vv03 = {-100, -1  , -100, -100, -100, -100, -100, -1  , -100, -1  , -100};     rewards.push_back(vv03);
    std::vector<int> vv04 = {-100, -1  , -1  , -1  , -1  , -1  , -1  , -1  , -100, -1  , -100};     rewards.push_back(vv04);
    std::vector<int> vv05 = {-100, -100, -100, -1  , -100, -100, -100, -1  , -100, -100, -100};     rewards.push_back(vv05);
    std::vector<int> vv06 = {-100, -1  , -1  , -1  , -1  , -1  , -1  , -1  , -1  , -1  , -100};     rewards.push_back(vv06);
    std::vector<int> vv07 = {-100, -100, -100, -100, -100, -1  , -100, -100, -100, -100, -100};     rewards.push_back(vv07);
    std::vector<int> vv08 = {-100, -1  , -1  , -1  , -1  , -1  , -1  , -1  , -1  , -1  , -100};     rewards.push_back(vv08);
    std::vector<int> vv09 = {-100, -100, -100, -1  , -100, -100, -100, -1  , -100, -100, -100};     rewards.push_back(vv09);
    std::vector<int> vv10 = {-100, -1  , -1  , -1  , -1  , -1  , -1  , -1  , -1  , -1  , -100};     rewards.push_back(vv10);
    std::vector<int> vv11 = {-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100};     rewards.push_back(vv11);

    /*
    for (int i=0; i<rewards.size(); i++){
        for (int j=0; j<rewards[i].size(); j++)
            std::cout << "\t" << rewards[i][j];
        std::cout << std::endl;
    }
    */
}


env_warehouse::~env_warehouse(){}



void env_warehouse::debug_mode(bool bDebug){
    this->bDebug = bDebug;
}



RowVector* env_warehouse::reset(){
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

    RowVector* obs = new RowVector(2);
    obs->coeffRef(0) = row;
    obs->coeffRef(1) = col;

    path_steps = 0;
    
    //std::vector<int> start_location = std::vector<int>{row,col};
    return obs;
}



std::vector<int> env_warehouse::get_env_dims(){
    return env_dims;
}


int env_warehouse::get_reward(int row, int col){
    return rewards[row][col];
}



void env_warehouse::render(){
    std::cout << "Location = (" << row << " " << col << ") " << path_steps + 1 << std::endl;

}



void env_warehouse::step(int action, RowVector &observation, float &reward, bool &done){
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

    observation.coeffRef(0) = row;
    observation.coeffRef(1) = col;
    reward = rewards[row][col];
    done = is_terminal_state();
}



bool env_warehouse::is_terminal_state(){
    // Determines if specified location is a terminal state
    if (rewards[row][col]==-1)
        return false;
    else
        return true;
}

