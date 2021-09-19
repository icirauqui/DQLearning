#include "DNN.hpp"



float f_activation(float x){
    if (f_act==1) //tanh
        return tanhf(x);
    else { // relu
        if (x>0) return x;
        else return 0;
    }
}

float f_activation_d(float x){
    if (f_act==1) //tanh
        return 1 - tanhf(x) * tanhf(x);
    else { // relu
        if (x>0) return 1;
        else return 0;
    }
}




DNN::DNN(std::vector<int> topology, float learningRate, bool bDebug){
    this->topology = topology;
    this->learningRate = learningRate;
    this->bDebug = bDebug;


    for (uint i=0; i<topology.size(); i++){
        // Initialize neuron layers
        if (i == topology.size()-1){
            neuronLayers.push_back(new Eigen::RowVectorXf(topology[i]));
            for (int j=0; j<neuronLayers.back()->size(); j++){
                neuronLayers.back()->coeffRef(j) = 0.0;
            }
        }
        else{
            neuronLayers.push_back(new Eigen::RowVectorXf(topology[i]+1));
            for (int j=0; j<neuronLayers.back()->size(); j++){
                neuronLayers.back()->coeffRef(j) = 0.0;
            }
            neuronLayers.back()->coeffRef(topology[i]) = 1.0;
        }
    }

    // Initialize cache and delta vectors
    for (int i=0; i<neuronLayers.size(); i++){
        cacheLayers.push_back(new Eigen::RowVectorXf(i+1));
        deltas.push_back(new Eigen::RowVectorXf(i+1));

    }

    // Initialize weights matrix
    for (int i=1; i<topology.size(); i++){
        if (i!=topology.size()-1){
            weights.push_back(new Eigen::MatrixXf(topology[i-1] + 1, topology[i]+1));
            weights.back()->setRandom();
            weights.back()->col(topology[i]).setZero();
            weights.back()->coeffRef(topology[i-1],topology[i]) = 1.0;
        }
        else{
            weights.push_back(new Eigen::MatrixXf(topology[i-1]+1,topology[i]));
            weights.back()->setRandom();
        }
    }

    if (bDebug) {
        std::cout << "DNN Topology = " << topology.size() << " ( ";
        for (int i=0; i<topology.size(); i++)
            std::cout << topology[i] << " ";
        std::cout << ")" << std::endl << std::endl;

        for (int i=0; i<neuronLayers.size(); i++){
            std::cout << "neuronLayers " << neuronLayers.size() << " - " << neuronLayers[i]->size() << " - ";
            for (int j=0; j<neuronLayers[i]->size(); j++)
                std::cout << neuronLayers[i]->coeffRef(j) << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl;

        for (int i=0; i<cacheLayers.size(); i++){
            std::cout << "cacheLayers " << cacheLayers.size() << " - " << cacheLayers[i]->size() << " - ";
            for (int j=0; j<cacheLayers[i]->size(); j++)
                std::cout << cacheLayers[i]->coeffRef(j) << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl;

        for (int i=0; i<deltas.size(); i++){
            std::cout << "deltas " << deltas.size() << " - " << deltas[i]->size() << " - ";
            for (int j=0; j<deltas[i]->size(); j++)
                std::cout << deltas[i]->coeffRef(j) << " ";
            std::cout << std::endl;
        }

        for (int i=0; i<weights.size(); i++){
            std::cout << std::endl << "weights " << weights.size() << " - " << i+1 << " - " << weights[i]->rows() << "/" << weights[i]->cols() << std::endl;
            for (int j=0; j<weights[i]->rows(); j++){
                for (int k=0; k<weights[i]->cols(); k++){
                    std::cout << " " << weights[i]->coeffRef(j,k);
                }
                std::cout << std::endl;
            }
        }
        std::cout << std::endl;
    }


}


DNN::~DNN(){}


void DNN::debug_mode(bool bDebug){
    this->bDebug = bDebug;
}




void DNN::forward(Eigen::RowVectorXf& input){
    // Set the input to input layer. Block(startRow, startCol, blockRows, blockCols) returns a part of the given matrix
    neuronLayers.front()->block(0, 0, 1, neuronLayers.front()->size()-1) = input;

    // Propagate the data forward
    for (uint i=1; i<topology.size(); i++) {
        (*neuronLayers[i]) = (*neuronLayers[i - 1]) * (*weights[i - 1]);
        if (i<topology.size()-1)
            neuronLayers[i]->block(0, 0, 1, topology[i]).unaryExpr(std::ptr_fun(f_activation));
    }
}

void DNN::forward(Eigen::RowVectorXf& input, Eigen::RowVectorXf& output){
    // Set the input to input layer. Block(startRow, startCol, blockRows, blockCols) returns a part of the given matrix
    neuronLayers.front()->block(0, 0, 1, neuronLayers.front()->size()-1) = input;

    // Propagate the data forward
    for (uint i=1; i<topology.size(); i++) {
        (*neuronLayers[i]) = (*neuronLayers[i - 1]) * (*weights[i - 1]);
        if (i<topology.size()-1)
            neuronLayers[i]->block(0, 0, 1, topology[i]).unaryExpr(std::ptr_fun(f_activation));
    }
    
    output = *neuronLayers.back();
}





void DNN::backward(Eigen::RowVectorXf& output){
    calcErrors(output);
    updateWeights();
}






void DNN::backward(Eigen::RowVectorXf& actions, Eigen::RowVectorXf& experimentals){
    // Calculate the errors made by neurons of last layer
    (*deltas.back()) = actions - experimentals;

    // Error calculation of hidden layers is different, we will begin by the last hidden
    // layer and we will continue till the first hidden layer
    for (uint i = topology.size() - 2; i>0; i--)
        (*deltas[i]) = (*deltas[i+1]) * (weights[i]->transpose());

    updateWeights();
}




void DNN::calcErrors(Eigen::RowVectorXf& output){
    // Calculate the errors made by neurons of last layer
    (*deltas.back()) = output - (*neuronLayers.back());

    // Error calculation of hidden layers is different, we will begin by the last hidden
    // layer and we will continue till the first hidden layer
    for (uint i = topology.size() - 2; i>0; i--)
        (*deltas[i]) = (*deltas[i+1]) * (weights[i]->transpose());
}





void DNN::updateWeights(){
    for (uint i=0; i<topology.size()-1; i++){
        // In this loop we are iterating over the different layers (from first hidden to output layer)
        // If this layer is the output layer, there is no bias neuron there, number of neurons specified = number of cols
        // If this layer is not the output layer, there is a bias neuron and number of neurons specified = number of cols - 1

        if (i != topology.size() - 2) {
            for (uint c = 0; c<weights[i]->cols() - 1; c++) {
                for (uint r = 0; r<weights[i]->rows(); r++) {
                    weights[i]->coeffRef(r,c) += learningRate * deltas[i+1]->coeffRef(c) * f_activation_d(cacheLayers[i+1]->coeffRef(c)) * neuronLayers[i]->coeffRef(r);
                }
            }
        }
        else {
            for (uint c = 0; c<weights[i]->cols(); c++) {
                for (uint r = 0; r<weights[i]->rows(); r++) {
                    weights[i]->coeffRef(r,c) += learningRate * deltas[i+1]->coeffRef(c) * f_activation_d(cacheLayers[i+1]->coeffRef(c)) * neuronLayers[i]->coeffRef(r);
                }
            }
        }
    }
}


void DNN::update_from_main(DNN *pDNN){
    weights = pDNN->weights;
}






