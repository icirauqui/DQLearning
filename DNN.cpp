#include "DNN.hpp"



Scalar f_activation(Scalar x){
    if (f_act==1) //tanh
        return tanhf(x);
    else { // relu
        if (x>0) return x;
        else return 0;
    }
}

Scalar f_activation_d(Scalar x){
    if (f_act==1) //tanh
        return 1 - tanhf(x) * tanhf(x);
    else { // relu
        if (x>0) return 1;
        else return 0;
    }
}





DNN::DNN(std::vector<uint> topology, Scalar learningRate){
    this->topology = topology;
    this->learningRate = learningRate;
    this->f_act = f_act;



    for (uint i=0; i<topology.size(); i++){
        // Initialize neuron layers
        if (i == topology.size()-1)
            neuronLayers.push_back(new RowVector(topology[i]));
        else
            neuronLayers.push_back(new RowVector(topology[i]+1));

        // Initialize cache and delta vectors
        cacheLayers.push_back(new RowVector(neuronLayers.size()));
        deltas.push_back(new RowVector(neuronLayers.size()));

        // vector.back() gives the handle to recently added element
        // coeffRef gives the reference of value at that place
        if (i != topology.size() - 1){
            neuronLayers.back()->coeffRef(topology[i]) = 1.0;
            cacheLayers.back()->coeffRef(topology[i]) = 1.0;
        }

        // Initialize weights matrix
        if (i>0){
            if (i!=topology.size()-1){
                weights.push_back(new Matrix(topology[i-1] + 1, topology[i]+1));
                weights.back()->setRandom();
                weights.back()->col(topology[i]).setZero();
                weights.back()->coeffRef(topology[i-1],topology[i]) = 1.0;
            }
            else{
                weights.push_back(new Matrix(topology[i-1]+1,topology[i]));
                weights.back()->setRandom();
            }
        }
    }







    if (bDebug) {
        std::cout << " neuron layers " << std::endl;
        for (int i=0; i<neuronLayers.size(); i++){
            RowVector aa = *neuronLayers[i];
            std::cout << aa.size() << " - ";
            for (int j=0; j<aa.size(); j++)
                std::cout << aa[j] << " ";
            std::cout << std::endl;
        }

        std::cout << " cache layers " << std::endl;
        for (int i=0; i<cacheLayers.size(); i++){
            RowVector aa = *cacheLayers[i];
            std::cout << aa.size() << " - ";
            for (int j=0; j<aa.size(); j++)
                std::cout << aa[j] << " ";
            std::cout << std::endl;
        }

        std::cout << " deltas " << std::endl;
        for (int i=0; i<deltas.size(); i++){
            RowVector aa = *deltas[i];
            std::cout << aa.size() << " - ";
            for (int j=0; j<aa.size(); j++)
                std::cout << aa[j] << " ";
            std::cout << std::endl;
        }

        std::cout << " weights " << std::endl;
        for (int i=0; i<weights.size(); i++){
            Matrix aa = *weights[i];
            std::cout << " - " << aa.size() << " " << aa.rows() << " " << aa.cols() << std::endl;
            for (int j=0; j<aa.rows(); j++){
                for (int k=0; k<aa.cols(); k++)
                    std::cout << aa(j,k) << " ";
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }
}


DNN::~DNN(){}


void DNN::debug_mode(bool bDebug){
    this->bDebug = bDebug;
}



/*
void DNN::propagateForward(RowVector& input){
    // Set the input to input layer. Block(startRow, startCol, blockRows, blockCols) returns a part of the given matrix
    neuronLayers.front()->block(0, 0, 1, neuronLayers.front()->size()-1) = input;

    // Propagate the data forward
    for (uint i=1; i<topology.size(); i++)
        (*neuronLayers[i]) = (*neuronLayers[i - 1]) * (*weights[i - 1]);

    // Apply the activation function to all elements of CURRENT_LAYER using unaryExpr
    for (uint i = 1; i < topology.size() - 1; i++)
        neuronLayers[i]->block(0, 0, 1, topology[i]).unaryExpr(std::ptr_fun(f_activation));
}
*/



/*
void DNN::propagateBackward(RowVector& output){
    calcErrors(output);
    updateWeights();
}
*/




/*
void DNN::propagateBackwardRL(RowVector& actions, RowVector& experimentals){
    // Calculate the errors made by neurons of last layer
    (*deltas.back()) = actions - experimentals;

    // Error calculation of hidden layers is different, we will begin by the last hidden
    // layer and we will continue till the first hidden layer
    for (uint i = topology.size() - 2; i>0; i--)
        (*deltas[i]) = (*deltas[i+1]) * (weights[i]->transpose());

    updateWeights();
}
*/




/*
void DNN::calcErrors(RowVector& output){
    // Calculate the errors made by neurons of last layer
    (*deltas.back()) = output - (*neuronLayers.back());

    // Error calculation of hidden layers is different, we will begin by the last hidden
    // layer and we will continue till the first hidden layer
    for (uint i = topology.size() - 2; i>0; i--)
        (*deltas[i]) = (*deltas[i+1]) * (weights[i]->transpose());
}
*/



/*
void DNN::updateWeights(){
    // topology.size()-1 = weights.size()
    for (uint i=0; i<topology.size()-1; i++){
        // In this loop we are iterating over the different layers (from first hidden to output layer)
        // If this layer is the output layer, there is no bias neuron there, number of neurons specified = number of cols
        // If this layer is not the output layer, there is a bias neuron and number of neurons specified = number of cols - 1
        if (i != topology.size() - 2) 
            for (uint c = 0; c<weights[i]->cols() - 1; c++)
                for (uint r = 0; r<weights[i]->rows(); r++)
                    weights[i]->coeffRef(r,c) += learningRate * deltas[i+1]->coeffRef(c) * f_activation_d(cacheLayers[i+1]->coeffRef(c)) * neuronLayers[i]->coeffRef(r);
        else
            for (uint c = 0; c<weights[i]->cols(); c++)
                for (uint r = 0; r<weights[i]->rows(); r++)
                    weights[i]->coeffRef(r,c) += learningRate * deltas[i+1]->coeffRef(c) * f_activation_d(cacheLayers[i+1]->coeffRef(c)) * neuronLayers[i]->coeffRef(r);
    }
}


void DNN::update_from_main(DNN *pDNN){
    weights = pDNN->weights;
}
*/


/*
int DNN::train_step(RowVector& input_data){
    propagateForward(*input_data);
    RowVector back = *neuronLayers.back();

    int max_action = 0;
    float max_q = 0.0;

    for (int i=0; i<back.size(); i++){
        if (back(i) > max_q) 
            max_action = i;
    }

    return max_action;
}
*/


/*
RowVector DNN::memory_step(RowVector& input_data){
    propagateForward(*input_data);
    RowVector back = *neuronLayers.back();
    return back;
}
*/

