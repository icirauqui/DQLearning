#include "nn1.hpp"

nn1::nn1(std::vector<uint> topology, Scalar learningRate, bool bDebug){
    this->topology = topology;
    this->learningRate = learningRate;
    this->bDebug = bDebug;

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
        // (using this as we're using pointers)
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
    


    //    for (int i=0; i<in_dat.size(); i++){
    //    std::cout << *in_dat[i] << " ";
    //}
    //std::cout << std::endl << std::endl << std::endl;



}


nn1::~nn1(){}


void nn1::propagateForward(RowVector& input){
    // Set the input to input layer. Block(startRow, startCol, blockRows, blockCols) returns a part of the given matrix
    neuronLayers.front()->block(0, 0, 1, neuronLayers.front()->size()-1) = input;

    // Propagate the data forward
    for (uint i=1; i<topology.size(); i++)
        (*neuronLayers[i]) = (*neuronLayers[i - 1]) * (*weights[i - 1]);

    // Apply the activation function to all elements of CURRENT_LAYER using unaryExpr
    for (uint i = 1; i < topology.size() - 1; i++)
        neuronLayers[i]->block(0, 0, 1, topology[i]).unaryExpr(std::ptr_fun(activationFunction));
}


void nn1::propagateBackward(RowVector& output){
    calcErrors(output);
    updateWeights();
}



void nn1::calcErrors(RowVector& output){
    // Calculate the errors made by neurons of last layer
    (*deltas.back()) = output - (*neuronLayers.back());

    // Error calculation of hidden layers is different, we will begin by the last hidden
    // layer and we will continue till the first hidden layer
    for (uint i = topology.size() - 2; i>0; i--)
        (*deltas[i]) = (*deltas[i+1]) * (weights[i]->transpose());
}



void nn1::updateWeights(){
    // topology.size()-1 = weights.size()
    for (uint i=0; i<topology.size()-1; i++){
        // In this loop we are iterating over the different layers (from first hidden to output layer)
        // If this layer is the output layer, there is no bias neuron there, number of neurons specified = number of cols
        // If this layer is not the output layer, there is a bias neuron and number of neurons specified = number of cols - 1
        if (i != topology.size() - 2) 
            for (uint c = 0; c<weights[i]->cols() - 1; c++)
                for (uint r = 0; r<weights[i]->rows(); r++)
                    weights[i]->coeffRef(r,c) += learningRate * deltas[i+1]->coeffRef(c) * activationFunctionDerivative(cacheLayers[i+1]->coeffRef(c)) * neuronLayers[i]->coeffRef(r);
        else
            for (uint c = 0; c<weights[i]->cols(); c++)
                for (uint r = 0; r<weights[i]->rows(); r++)
                    weights[i]->coeffRef(r,c) += learningRate * deltas[i+1]->coeffRef(c) * activationFunctionDerivative(cacheLayers[i+1]->coeffRef(c)) * neuronLayers[i]->coeffRef(r);
    }
}



void nn1::train(std::vector<RowVector*> input_data, std::vector<RowVector*> output_data){
    for (uint i = 0; i < input_data.size(); i++){
        if (bDebug) std::cout << "Input-Expected-Computed-MSE\t" << *input_data[i];
        propagateForward(*input_data[i]);
        if (bDebug) std::cout << "\t" << *output_data[i] << "\t" << *neuronLayers.back();
        propagateBackward(*output_data[i]);
        if (bDebug) std::cout << "\t" << std::sqrt((*deltas.back()).dot((*deltas.back())) / deltas.back()->size()) << std::endl;
    }
}


Scalar nn1::activationFunction(Scalar x){
    return tanhf(x);
}

Scalar nn1::activationFunctionDerivative(Scalar x){
    return 1 - tanhf(x) * tanhf(x);
}

