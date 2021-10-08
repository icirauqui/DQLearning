#include <iostream>

#include "agentDQL.cpp"

int main(){

    agentDQL* pAgent = new agentDQL({4,50,50,2},0.5,0.5,1.0);
    
    int episodes = 100000;

    int option = 0;



    do {
        std::cout << " Menu : " << std::endl
                  << "  1. Train " << std::endl
                  << "  2. Test  " << std::endl
                  << "  3. Save  " << std::endl
                  << "  4. Load  " << std::endl
                  << " " << std::endl
                  << " Opt: ";
        std::cin >> option;

        if (option==1){
            pAgent->debug_mode(false);
            pAgent->train(episodes,500,100,10);
        }
        else if (option==2){
            pAgent->debug_mode(true);
            pAgent->train(10,500,100,10);
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





