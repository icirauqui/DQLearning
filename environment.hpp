#include <iostream>
#include <vector>
#include <fstream>

class environment{
    private:

        // Create a 2D array to hold the rewards for each state
        // The array is 11x11, matching the shape of the environment
        std::vector<std::vector<int> > rewards;

    
        bool bDebug = false;

        std::vector<int> env_dims;


    public:
        environment(std::vector<int> env_dims);
        ~environment();

        void debug(bool bDebug);

        void reset();

        std::vector<int> get_env_dims();

        




};
