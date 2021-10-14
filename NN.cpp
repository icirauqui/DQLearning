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
    //std::cout << "model size = " << model.size() << std::endl;

/*
    std::vector<std::vector<float> > layer0 = std::vector<std::vector<float> > (5, std::vector<float>(24,0));
    std::vector<std::vector<float> > layer1 = std::vector<std::vector<float> > (25, std::vector<float>(24,0));
    std::vector<std::vector<float> > layer2 = std::vector<std::vector<float> > (25, std::vector<float>(2,0));

    std::ifstream fp0("layer_0.csv");
    for (unsigned int i=0; i<layer0.size(); i++)
        for (unsigned int j=0; j<layer0[i].size(); j++)
            fp0 >> layer0[i][j];
    std::ifstream fp1("layer_1.csv");
    for (unsigned int i=0; i<layer1.size(); i++)
        for (unsigned int j=0; j<layer1[i].size(); j++)
            fp1 >> layer1[i][j];
    std::ifstream fp2("layer_2.csv");
    for (unsigned int i=0; i<layer2.size(); i++)
        for (unsigned int j=0; j<layer2[i].size(); j++)
            fp2 >> layer2[i][j];
    

    Eigen::MatrixXf* weights0 = new Eigen::MatrixXf(layer0.size(),layer0[0].size());
    Eigen::MatrixXf* weights1 = new Eigen::MatrixXf(layer1.size(),layer1[0].size());
    Eigen::MatrixXf* weights2 = new Eigen::MatrixXf(layer2.size(),layer2[0].size());

    for (int i=0; i<layer0.size(); i++)
        for (int j=0; j<layer0[i].size(); j++)
            weights0->coeffRef(i,j) = layer0[i][j];
    for (int i=0; i<layer1.size(); i++)
        for (int j=0; j<layer1[i].size(); j++)
            weights1->coeffRef(i,j) = layer1[i][j];
    for (int i=0; i<layer2.size(); i++)
        for (int j=0; j<layer2[i].size(); j++)
            weights2->coeffRef(i,j) = layer2[i][j];

    model[0]->set_weights(weights0);
    model[1]->set_weights(weights1);
    model[2]->set_weights(weights2);

    for (int i=0; i<model.size(); i++){
        Eigen::MatrixXf* weights = model[i]->get_weights();
        std::cout << std::endl << std::endl << "Weights " << i << std::endl;
        for (int r=0; r<weights->rows(); r++){
            for (int c=0; c<weights->cols(); c++){
                std::cout << " " << weights->coeffRef(r,c);
            }
            std::cout << std::endl;
        }
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
    
    for (unsigned int i=0; i<model.size(); i++){
        //std::cout << " A(" << i+1 << "/" << model.size() << ")-1 " << std::endl;
        vals = model[i]->forward(*vals);
        //std::cout << " A(" << i+1 << "/" << model.size() << ")-2 " << std::endl;
    }

    output = *vals;
}



void NN::backward(Eigen::RowVectorXf& actions, Eigen::RowVectorXf& experimentals){
    Eigen::RowVectorXf* delta = new Eigen::RowVectorXf(actions.size());
    (*delta) = actions - experimentals;

    //std::cout << std::endl << std::endl;
    //std::cout << "Delta 0 = " << *delta << std::endl;
    int j = 1;
    for (int i=model.size()-1; i>=0; i--){
        //std::cout << std::endl;
        //std::cout << i << " backward[" << i << "]" << std::endl;
        delta = model[i]->backward(*delta);
        //std::cout << "Delta " << j << " = " << *delta << std::endl;
        j++;
        //std::cout << i << " backward[" << i << "] done" << std::endl;
    }
}


void NN::update_time(){
    for (int i=0; i<model.size(); i++)
        model[i]->update_time();
}