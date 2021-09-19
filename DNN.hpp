#ifndef DNN_HPP
#define DNN_HPP


#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <vector>


// Activation functions and derivatives: 1(hyperbolic tangent) 2(ReLu)
int f_act = 2;
float f_activation(float x);
float f_activation_d(float x);


class DNN{

    public:

        // Constructor, desctructor, modes
        DNN(std::vector<int> topology, float learningRate = float(0.005), bool bDebug = false);
        ~DNN();
        void debug_mode(bool bDebug = false);

        // Forward propagation of data
        void forward(Eigen::RowVectorXf& input);
        void forward(Eigen::RowVectorXf& input, Eigen::RowVectorXf& output);

        // Backward propagation of errors made by neurons
        void backward(Eigen::RowVectorXf& output);
        void backward(Eigen::RowVectorXf& actions, Eigen::RowVectorXf& experimentals);

        // Calculate errors made by neurons in each layer
        void calcErrors(Eigen::RowVectorXf& output);

        // Update the weights of connections
        void updateWeights();
        void update_from_main(DNN *pDNN);


        /*
         * Storage objects for working of neural network
         * Use pointers when using std::vector<Class>, otherwise destructor of Class is called as it is pushed back!
         * Besides it also makes the NN class less heavy!! Try moving to Smart Pointers. 
         */
        std::vector<Eigen::RowVectorXf*> neuronLayers;   // Stores the different layers of our network, each one is an array of neurons
                                                // We store each layer in a vector, each element stores the activation value of the neuron

        std::vector<Eigen::RowVectorXf*> cacheLayers;    // Stores the unactivated values of layers
        std::vector<Eigen::RowVectorXf*> deltas;         // Stores the error contribution of each neuron
        std::vector<Eigen::MatrixXf*> weights;           // The weight of each connection.
        float learningRate;        


        // Topology describes how many neurons we have in each layer, its size is the number of layers.
        std::vector<int> topology;

        // Debugging flag
        bool bDebug = false;



};

#endif