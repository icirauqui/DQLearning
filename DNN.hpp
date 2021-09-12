#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <vector>


typedef float Scalar;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;


class DNN{

    public:

        // Constructor
        DNN(std::vector<uint> topology, Scalar learningRate = Scalar(0.005));
        
        // Destructor
        ~DNN();

        void debug_mode(bool bDebug = false);

        // Function for forward propagation of data
        void propagateForward(RowVector& input);

        // Function for backward propagation of errors made by neurons
        void propagateBackward(RowVector& output);

        // Function to calculate errors made by neurons in each layer
        void calcErrors(RowVector& output);

        // Function to update the weights of connections
        void updateWeights();

        // Function to train the neural network give an array of datapoints
        void train(std::vector<RowVector*> input_data, std::vector<RowVector*> output_data);


        // Activation functions and derivatives: 1(hyperbolic tangent) 2(ReLu)
        static Scalar f_activation(Scalar x);
        static Scalar f_activation_d(Scalar x);

        static const int f_act = 2;



        /*
         * Storage objects for working of neural network
         * Use pointers when using std::vector<Class>, otherwise destructor of Class is called as it is pushed back!
         * Besides it also makes the NN class less heavy!! Try moving to Smart Pointers. 
         */
        std::vector<RowVector*> neuronLayers;   // Stores the different layers of our network, each one is an array of neurons
                                                // We store each layer in a vector, each element stores the activation value of the neuron

        std::vector<RowVector*> cacheLayers;    // Stores the unactivated values of layers
        std::vector<RowVector*> deltas;         // Stores the error contribution of each neuron
        std::vector<Matrix*> weights;           // The weight of each connection.
        Scalar learningRate;        


        // Topology describes how many neurons we have in each layer, its size is the number of layers.
        std::vector<uint> topology;

        // Debugging flag
        bool bDebug = false;



};