#include <iostream>


#include "agentDQL.cpp"




int main(){

    agentDQL* pAgent = new agentDQL({11,11},{10,10,4},0.5,0.5,1.0);

    int opt = 0;
    int episodes = 0;
    while (opt != 9){
        std::cout << std::endl
                  << " Menu " << std::endl 
                  << "   1. Train " << std::endl
                  << "   2. Train 1000 episodes " << std::endl
                  << "   3. Tests " << std::endl
                  << "   4. Offline " << std::endl
                  << "   9. Quit" << std::endl
                  << std::endl 
                  << " Option ";
        std::cin >> opt;

        if (opt==1){
            std::cout << " Episodes = ";
            std::cin >> episodes;
            pAgent->train(episodes,1000,100,4);
        }

        else if (opt==2){
            std::cout << " Train 1000 episodes" << std::endl;
            pAgent->train(1000,1000,100,4);
        }

        else if (opt==3){
   
            Eigen::RowVectorXf* observation = new RowVector(2);
            Eigen::RowVectorXf* observation1 = new RowVector(2);

            observation->coeffRef(0) = 1;
            observation->coeffRef(1) = 2;

            for (int i=0; i<2; i++)
                std::cout << observation->coeffRef(i) << " " << observation1->coeffRef(i) << std::endl;

            *observation1 = *observation;

            std::cout << std::endl;
            for (int i=0; i<2; i++)
                std::cout << observation->coeffRef(i) << " " << observation1->coeffRef(i) << std::endl;

            observation->coeffRef(0) = 3;
            observation->coeffRef(1) = 4;

            std::cout << std::endl;
            for (int i=0; i<2; i++)
                std::cout << observation->coeffRef(i) << " " << observation1->coeffRef(i) << std::endl;

        }

    }

    
 
    
 
    return 0;
    
}



