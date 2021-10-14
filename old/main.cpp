#include <iostream>

#include "agentDQL.cpp"

int main(){

    agentDQL* pAgent = new agentDQL({4,24,24,2},0.001,0.95,1.0);
    
    float r3;
    float LO = -0.05;
    float HI = +0.05;

    for (int i=0; i<10; i++){
        r3 = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
        std::cout << " r3 = " << r3 << std::endl;
    }


    int episodes = 1000;

    int option = 0;

    do {
        std::cout << " Menu : " << std::endl
                  << "  1. Train " << std::endl
                  << "  2. Test  " << std::endl
                  << "  3. Save  " << std::endl
                  << "  4. Load  " << std::endl
                  << " .9. Exit  " << std::endl
                  << " " << std::endl
                  << " Opt: ";
        std::cin >> option;

        if (option==1){
            std::cout << "Episodes = "; std::cin >> episodes;
            pAgent->debug_mode(false);
            pAgent->train(episodes,500,100,20);
        }
        else if (option==2){
            pAgent->test(500,10,true);
            //pAgent->debug_mode(true);
            //pAgent->train(10,500,100,10);
        }
    } while (option != 9);

    //pAgent->train(episodes,500,100,10);

    //pAgent->backup_epsilon();
    //pAgent->debug_mode(true);
    //pAgent->train(1,1000,100,4);
    //pAgent->debug_mode(false);
    //pAgent->restore_epsilon();

    return 0;















































}









