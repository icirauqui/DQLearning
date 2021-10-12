#include "NN.hpp"

NN::NN(std::vector<int> topology, float learningRate, bool bDebug){
    this->topology = topology;
    this->learningRate = learningRate;
    this->bDebug = bDebug;


    for (unsigned int i=0; i<topology.size()-1; i++)
        model.push_back(new NNLayer(topology[i]+1,topology[i+1]));
}


NN::~NN(){}

void NN::debug_mode(bool bdbg){
    this->bDebug = bdbg;
}



void NN::forward(Eigen::RowVectorXf& input, Eigen::RowVectorXf& output){
    Eigen::RowVectorXf* vals = new Eigen::RowVectorXf(input.size());
    *vals = input;
    
    for (unsigned int i=0; i<model.size(); i++){
        //std::cout << " A(" << i+1 << "/" << model.size() << ")-1 " << std::endl;
        vals = model[i]->forward(*vals);
        //std::cout << " A(" << i+1 << "/" << model.size() << ")-2 " << std::endl;
    }

    output = *vals;
}



void NN::backward(Eigen::RowVectorXf& actions, Eigen::RowVectorXf& experimentals){
    Eigen::RowVectorXf* delta = new Eigen::RowVectorXf(actions.size());
    *delta = actions - experimentals;

    std::cout << "Delta = " << *delta << std::endl;
    for (unsigned int i=model.size()-1; i>=0; i--){
        std::cout << i << " backward[" << i+1-(model.size()-1) << "/" << model.size() << "]" << std::endl;
        delta = model[i]->backward(*delta);
        std::cout << i << " backward[" << i+1-(model.size()-1) << "/" << model.size() << "] done" << std::endl;
    }
}