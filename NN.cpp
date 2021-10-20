#include "NN.hpp"

NN::NN(std::vector<int> topology, std::vector<std::string> act_funcs1, float learningRate, bool bDebug){
    this->topology = topology;
    this->learningRate = learningRate;
    this->bDebug = bDebug;

    this->act_funcs = act_funcs;
    for (int i=0; i<act_funcs1.size(); i++)
        this->act_funcs.push_back(act_funcs1[i]);

    for (unsigned int i=0; i<topology.size()-1; i++)
        model.push_back(new NNLayer(topology[i]+1,topology[i+1],act_funcs[i],learningRate));

    for (int i=0; i<topology.size(); i++)
        std::cout << topology[i] << " ";
    std::cout << std::endl;
    
    for (int i=0; i<act_funcs.size(); i++)
        std::cout << act_funcs[i] << " ";
    std::cout << std::endl;
    

/*
    for (unsigned int i=0; i<topology.size()-1; i++){
        if (i<topology.size()-2)
            model.push_back(new NNLayer(topology[i]+1,topology[i+1],"relu",learningRate));
        else
            model.push_back(new NNLayer(topology[i]+1,topology[i+1],"none",learningRate));
    }
*/

}


NN::~NN(){}

void NN::debug_mode(bool bdbg){
    this->bDebug = bdbg;
}



void NN::forward(Eigen::RowVectorXf& input, Eigen::RowVectorXf& output){
    Eigen::RowVectorXf* vals = new Eigen::RowVectorXf(input.size());
    *vals = input;
    
    for (unsigned int i=0; i<model.size(); i++)
        vals = model[i]->forward(*vals);

    output = *vals;
}



void NN::backward(Eigen::RowVectorXf& actions, Eigen::RowVectorXf& experimentals){
    Eigen::RowVectorXf* delta = new Eigen::RowVectorXf(actions.size());
    (*delta) = actions - experimentals;

    for (int i=model.size()-1; i>=0; i--)
        delta = model[i]->backward(*delta);
}


void NN::update_time(){
    for (int i=0; i<model.size(); i++)
        model[i]->update_time();
}