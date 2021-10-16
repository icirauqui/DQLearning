#include "agentDQL.hpp"



agentDQL::agentDQL(std::vector<int> topology1, float learningRate, float gamma, float epsilon){
    this->learning_rate = learning_rate;
    this->gamma = gamma;
    this->epsilon = epsilon;
    this->epsilon1 = epsilon;
    this->topology = topology1;

    pEnv = new env_cart_pole();
    pMemory = new memory_buffer();

    pNN1 = new NN(topology,learningRate,false);
    pNN2 = new NN(topology,learningRate,false);

    for (int i=0; i<pNN1->model.size(); i++)
        pNN2->model[i]->set_weights(pNN1->model[i]->get_weights());

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
    float reward = 0.0;
    float ep_reward = 0.0;
    float max_ep_reward = -1000;
    bool done;
    
    std::vector<float> last_100_ep_rewards;

    for (int episode=0, cnt_target_upd = 0, cnt_exp_upd = 0; episode<num_episodes; episode++){
        
        observation = pEnv->reset();
        
        last_100_ep_rewards.push_back(ep_reward);
        if (last_100_ep_rewards.size()>100)
            last_100_ep_rewards.erase(last_100_ep_rewards.begin());

        max_ep_reward = 0.0;
        for (int i=0; i<last_100_ep_rewards.size(); i++)
            max_ep_reward += last_100_ep_rewards[i];
        max_ep_reward /= last_100_ep_rewards.size();
        
        ep_reward = 0.0;

        for (int step = 0; step < max_steps; step++, cnt_target_upd++, cnt_exp_upd++){
            // Render environment, print state
            if (bDebug)
                pEnv->render();

            // Select action with main NN, performs a forward that returns Q(s,a;th)
            int action = this->select_epsilon_greedy_action(*observation,true);

            *observation1 = *observation;

            // Apply action to get new state, check if we are at a terminal state.
            pEnv->step(action, *observation, reward, done);

            // Store the occurence for future learning
            pMemory->add(*observation, *observation1, action, done);

            // Add to total episode reward
            ep_reward += reward;
            
            // Learn from past outcomes every n steps
            if (cnt_exp_upd == exp_upd){
                //std::cout << "exp_upd" << std::endl;
                if(exp_upd < pMemory->size())
                    this->experience_replay(exp_upd);
                pNN1->update_time();
                pNN2->update_time();
                cnt_exp_upd = 0;
            }

            // Update target NN weights every m steps (m>n)
            if (cnt_target_upd == target_upd){
                for (int i=0; i<pNN1->model.size(); i++)
                    pNN2->model[i]->set_weights(pNN1->model[i]->get_weights());
                cnt_target_upd = 0;
            }
            

            // End episode if we have reached a terminal state
            if (done){
                if (bDebug)
                    pEnv->render();
                    std::cout << "Episode " << episode << " (" << 100*episode/num_episodes << "%) has ended after " << step + 1 << " with reward " << ep_reward << "/" << max_ep_reward << " " << epsilon << std::endl;
                break;
            }
        }
    }
}


void agentDQL::test(int max_steps, int num_episodes, bool verbose){

    RowVector* observation = new RowVector(2);
    int action;
    float reward, ep_reward;
    float max_ep_reward = -1000;
    bool done;
    
    std::vector<float> last_100_ep_rewards;

    for (int episode=0, cnt_target_upd = 0, cnt_exp_upd = 0; episode<num_episodes; episode++){
        
        observation = pEnv->reset();
        
        last_100_ep_rewards.push_back(ep_reward);
        if (last_100_ep_rewards.size()>100)
            last_100_ep_rewards.erase(last_100_ep_rewards.begin());

        max_ep_reward = 0.0;
        for (int i=0; i<last_100_ep_rewards.size(); i++)
            max_ep_reward += last_100_ep_rewards[i];
        max_ep_reward /= 100;

        ep_reward = 0.0;

        for (int step = 0; step < max_steps; step++, cnt_target_upd++, cnt_exp_upd++){
            if (verbose)
                pEnv->render();

            int action = this->select_epsilon_greedy_action(*observation, false);

            pEnv->step(action, *observation, reward, done);

            ep_reward += reward;

            if (done){
                if (verbose)
                    pEnv->render();
                std::cout << "Episode " << episode << " has ended after " << step + 1 << " with reward " << ep_reward << "/" << max_ep_reward << " " << epsilon << std::endl;
                break;
            }
        }
    }
}





int agentDQL::select_epsilon_greedy_action(RowVector& obs, bool bTrain){
    // Calculate next action, either randomly, or with Main NN
    float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    if (bTrain && r<epsilon){
        return floor(r*(topology.back()-0.01));
    }
    else {
        RowVector output;
        pNN1->forward(obs,output);
        float max_q = - static_cast <float> (RAND_MAX);
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
        this->epsilon *= 0.9999;
}



void agentDQL::experience_replay(int update_size){
    int memory_size = pMemory->size();

    for (int i=0; i<update_size; i++){
        int idx = rand() % memory_size;

        RowVector* new_obs = pMemory->sample_observation(idx);
        RowVector* prev_obs = pMemory->sample_observation1(idx);
        int action_selected = pMemory->sample_action(idx);
        bool done = pMemory->sample_done(idx);

        RowVector action_values(topology.back()), next_action_values(topology.back()), experimental_values(topology.back());
        pNN1->forward(*prev_obs,action_values);
        pNN2->forward(*new_obs,next_action_values);
        experimental_values = action_values;

        float max_next_action_values = - static_cast <float> (RAND_MAX);
        for (int i=0; i<next_action_values.size(); i++){
            if (next_action_values(i) > max_next_action_values) 
                max_next_action_values = next_action_values(i);
        }

        if (done)
            experimental_values[action_selected] = -1;
        else
            experimental_values[action_selected] = 1 + gamma*max_next_action_values;

        pNN1->backward(action_values, experimental_values);
    }

    epsilon_decay();
}


void agentDQL::backup_epsilon(){
    epsilon1 = epsilon;
    epsilon = 0.0;
}

void agentDQL::restore_epsilon(){
    epsilon = epsilon1;
}


void agentDQL::save_model(){
    std::vector<Eigen::MatrixXf*> model_weigths;
    for (int i=0; i<pNN1->model.size(); i++)
        model_weigths.push_back(pNN1->model[i]->get_weights());
    for (int i=0; i<pNN2->model.size(); i++)
        model_weigths.push_back(pNN2->model[i]->get_weights());

    std::string fileName = "model.csv";
    std::ofstream fileOut;
    fileOut.open(fileName);

    for (int i=0; i<model_weigths.size(); i++){
        for (int r=0; r<model_weigths[i]->rows(); r++)
            for (int c=0; c<model_weigths[i]->cols(); c++)
                fileOut << model_weigths[i]->coeffRef(r,c) << " ";
        fileOut << std::endl;
    }
    fileOut.close();

    std::cout << "Model Saved" << std::endl << std::endl;
}


void agentDQL::load_model(){
    std::vector<Eigen::MatrixXf*> model_weigths;
    for (int i=0; i<pNN1->model.size(); i++)
        model_weigths.push_back(pNN1->model[i]->get_weights());
    for (int i=0; i<pNN2->model.size(); i++)
        model_weigths.push_back(pNN2->model[i]->get_weights());

    std::string fileName = "model.csv";
    std::ifstream fileIn;

    for (int i = 0; i<model_weigths.size(); i++){      
        for (int r=0; r<model_weigths[i]->rows(); r++)
            for (int c=0; c<model_weigths[i]->cols(); c++)
                fileIn >> model_weigths[i]->coeffRef(r,c);
    }

    for (int i=0; i<model_weigths.size(); i++){
        int model_size = pNN1->model.size();
        if (i<model_size)
            pNN1->model[i]->set_weights(model_weigths[i]);
        else
            pNN2->model[i-model_size]->set_weights(model_weigths[i]);
    }

    std::cout << "Model Loaded" << std::endl << std::endl;
}