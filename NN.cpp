#include "NN.hpp"

NN::NN(std::vector<int> topology, float learningRate, bool bDebug){
    this->topology = topology;
    this->learningRate = learningRate;
    this->bDebug = bDebug;


    for (unsigned int i=0; i<topology.size()-1; i++){
        if (i<topology.size()-2)
            model.push_back(new NNLayer(topology[i]+1,topology[i+1],"relu",learningRate));
        else
            model.push_back(new NNLayer(topology[i]+1,topology[i+1],"none",learningRate));
    }
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