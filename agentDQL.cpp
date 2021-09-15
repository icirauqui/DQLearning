#include "agentDQL.hpp"






agentDQL::agentDQL(environment* pEnv, std::vector<uint> topology, float learningRate, float discount_factor, float epsilon){
    
    this->pEnv = pEnv;
    this->learning_rate = learning_rate;
    this->discount_factor = discount_factor;
    this->epsilon = epsilon;

    std::vector<int> env_dims = pEnv->get_env_dims();
    this->topology.push_back(env_dims.size());
    for (int i=0; i<topology.size(); i++)
        this->topology.push_back(topology[i]);

    pDNN1 = new DNN(this->topology,learningRate);
    pDNN2 = new DNN(this->topology,learningRate);

    // Ready randon number generator
    srand(time(0));
}


agentDQL::~agentDQL(){}




void agentDQL::debug_mode(bool bDebug){
    this->bDebug = bDebug;
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


void agentDQL::get_starting_location(){
    // Determines a random not terminal starting location    
    int rowt = 0;
    int colt = 0;
            
    do {
        rowt = rand() % 11;
        colt = rand() % 11;
    }
    while (rewards[rowt][colt] != -1);

    row = rowt;
    col = colt;
};


//void get_next_action(bool exploration = true);
//void get_next_location();










void agentDQL::train(int num_episodes, int max_steps, int target_upd, int exp_upd){

    int cnt_target_upd = 0;
    int cnt_exp_upd = 0;

    for (int episode=0; episode<num_episodes; episode++){

        RowVector* observation = new RowVector(row,col);

        for (unsigned int step=0; step<max_steps; step++){




        }

        // Every n steps, compy weights from Main NN to Target NN
        if (i%update_target_episodes==0){
            pDNN2->update_from_main(pDNN1);
        }

        // Calculate next action, either randomly, or with Main NN
        int action = 0;
        float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        if (r<epsilon){
            RowVector* rv = new RowVector{row, col};
            action = pDNN1->train_step(rv);
        }
        else{
            action = rand() % pDNN1->topology[0];
        }


        //get_next_location();



    }



}


