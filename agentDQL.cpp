#include "agentDQL.hpp"






agentDQL::agentDQL(std::vector<uint> topology, float learningRate, float discount_factor, float epsilon){
    this->learning_rate = learning_rate;
    this->discount_factor = discount_factor;
    this->epsilon = epsilon;

    pDNN1 = new DNN(topology,learningRate);
    pDNN2 = new DNN(topology,learningRate);
}


agentDQL::~agentDQL(){}




void agentDQL::debug_mode(bool bDebug){
    this->bDebug = bDebug;
}


void agentDQL::train(int train_episodes, int update_target){
    for (int i=0; i<train_episodes; i++){
        if (i%update_target==0){
            pDNN2->update_from_main(pDNN1);
        }
        else{
            RowVector* rv = new RowVector{row, col};
            action = pDNN1->train_step(rv);
            get_next_location();
        }
    }



}



/*
bool agentDQL::is_terminal_state(){
    // Determines if specified location is a terminal state
    if (rewards[row][col]==-1)
        return false;
    else
        return true;
}
*/