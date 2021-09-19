#include "agentDQL.hpp"






agentDQL::agentDQL(std::vector<int> env_dims, std::vector<int> topology1, float learningRate, float discount_factor, float epsilon){
    this->learning_rate = learning_rate;
    this->discount_factor = discount_factor;
    this->epsilon = epsilon;

    pEnv = new environment(env_dims);
    pMemory = new memory_buffer();

    topology.push_back(env_dims.size());
    for (unsigned int i=0; i<topology1.size(); i++)
        topology.push_back(topology1[i]);

    pDNN1 = new DNN(topology,learningRate,false);
    pDNN2 = new DNN(topology,learningRate,false);

    // Ready randon number generator
    srand(time(0));
}


agentDQL::~agentDQL(){}



void agentDQL::debug_mode(bool bDebug){
    this->bDebug = bDebug;
}





void agentDQL::train(int num_episodes, int max_steps, int target_upd, int exp_upd){

    RowVector* observation = new RowVector(2);
    RowVector* observation1 = new RowVector(2);
    int action;
    float reward;
    bool done;
    
    std::vector<float> last_100_ep_rewards;

    for (int episode=0, cnt_target_upd = 0, cnt_exp_upd = 0; episode<num_episodes; episode++){
        
        observation = pEnv->reset();

        for (int step = 0; step < max_steps; step++, cnt_target_upd++, cnt_exp_upd++){
            
            pEnv->render();

            int action = this->select_epsilon_greedy_action(*observation);

            *observation1 = *observation;

            pEnv->step(*observation, reward, done, action);
            
            pMemory->add(*observation, *observation1, action, done);

            // Learn from past outcomes
            if (cnt_exp_upd == exp_upd){
                std::cout << "exp_upd" << std::endl;
                if(5*exp_upd < pMemory->size())
                    this->experience_replay(5*exp_upd);
                cnt_exp_upd = 0;
            }

            // Update target NN weights
            if (cnt_target_upd == target_upd){
                std::cout << "target_upd " << std::endl;
                pDNN2->update_from_main(pDNN1);
                cnt_target_upd = 0;
            }

            // End episode if we have reached a terminal state
            if (done){
                pEnv->render();
                std::cout << "Episode " << episode << " has ended after " << step + 1 << std::endl;
                break;
            }
        }
    }
}





int agentDQL::select_epsilon_greedy_action(RowVector& obs){
    // Calculate next action, either randomly, or with Main NN
    float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

    if (r>epsilon){
        return rand() % topology.back();
    }
    else {
        RowVector output;
        pDNN1->forward(obs,output);
        float max_q = 0.0;
        int act = 0;
        for (int i=0; i<output.size(); i++){
            if (output[i]>max_q){
                max_q = output[i];
                act = i;
            }
        }
        return act;
    }
}




void agentDQL::epsilon_decay(){
    if (this->epsilon > 0.001)
        this->epsilon *= 0.999;
}



void agentDQL::experience_replay(int update_size){
    int memory_size = pMemory->size();
    for (int i=0; i<update_size; i++){
        int idx = rand() % memory_size;

        RowVector* new_obs = pMemory->sample_observation(idx);
        RowVector* prev_obs = pMemory->sample_observation1(idx);
        int action_selected = pMemory->sample_action(idx);
        bool done = pMemory->sample_done(idx);

        //std::cout << "new_obs ";
        //for (int it1=0; it1<new_obs->size(); it1++)
        //    std::cout << new_obs->coeffRef(it1) << " ";
        //std::cout << std::endl;

        //std::cout << "prev_obs ";
        //for (int it1=0; it1<prev_obs->size(); it1++)
        //    std::cout << prev_obs->coeffRef(it1) << " ";
        //std::cout << std::endl;

        RowVector action_values(4), next_action_values(4), experimental_values(4);
        pDNN1->forward(*prev_obs,action_values);
        pDNN1->forward(*new_obs,next_action_values);
        experimental_values = action_values;

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

        pDNN1->backward(action_values, experimental_values);
    }

    // Epsilon decays as learning advances
    epsilon_decay();
}
