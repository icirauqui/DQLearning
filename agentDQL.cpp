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
void agentDQL::train(int num_episodes, int max_steps, int target_upd, int exp_upd){

    for (int episode=0; episode<num_episodes; episode++){

        RowVector* observation = pEnv->reset();
        RowVector* observation1 = pEnv->reset();

        for (int step = 0, cnt_target_upd = 0, cnt_exp_upd = 0; step < max_steps; step++, cnt_target_upd++, cnt_exp_upd++){

            pEnv->render();

            this->action = this->select_action(*observation);

            for (int i=0; i<observation->size(); i++)
                observation1->coeffRef(i) = observation->coeffRef(i);

            float reward;
            bool done;
            pEnv->step(&observation, &reward, &done, action);
            
            // Store result for further learning
            remember(observation, observation1, action, done);

            // Learn from past outcomes
            if (cnt_exp_upd == exp_upd){
                this->experience(5*exp_upd);
                cnt_exp_upd = 0;
            }

            if (cnt_target_upd == target_upd){
                // Every n steps, compy weights from Main NN to Target NN
                pDNN2->update_from_main(pDNN1);
                cnt_target_upd = 0;
            }



            if (done){
                // If we have reached a terminal location, end this episode
                std::cout << "Episode " << episode << " has ended after " << step << std::endl;
                // print(model.layers[0].lr)
                break;
            }
        }
    }
}
*/


/*
int agentDQL::select_action(RowVector& obs){
    // Calculate next action, either randomly, or with Main NN

    int act = 0;
    float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

    if (r<epsilon)
        act = pDNN1->train_step(*obs);
    else
        act = rand() % pDNN1->topology[0];

    return act;
}
*/


void agentDQL::epsilon_decay(){
    if (this->epsilon > 0.001)
        this->epsilon *= 0.999;
}

/*
void agentDQL::remember(RowVector& obs, RowVector& obs1, int act, bool bdone){
    memory_observation.push_back(*obs);
    memory_observation1.push_back(*obs1);
    memory_action.push_back(act);
    memory_done.push_back(bdone);
}
*/

/*
void agentDQL::experience(int update_size){

    for (int i=0; i<update_size; i++){
        int idx = rand() % memory_done.size();

        RowVector* new_obs = memory_observation[idx];
        RowVector* prev_obs = memory_observation1[idx];
        int action_selected = memory_action[idx];
        bool done = memory_done[idx];

        RowVector action_values = pDNN1->memory_step(*prev_obs);
        RowVector next_action_values = pDNN1->memory_step(*new_obs);
        RowVector experimental_values = action_values;


        float max_next_action_values = 0.0;
        for (int i=0; i<next_action_values.size(); i++){
            if (next_action_values(i) > max_next_action_values) 
                max_next_action_values = next_action_values(i);
        }

        if (done){
            experimental_values[action_selected] = -1;
        }
        else{
            experimental_values[action_selected] = 1 + discount_factor*max_next_action_values;
        }

        pDNN1->propagateBackwardRL(action_values, experimental_values);
    }

    // Epsilon decays as learning advances
    epsilon_decay();
}
*/