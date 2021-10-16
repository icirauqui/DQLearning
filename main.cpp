#include <iostream>

#include "agentDQL.cpp"

int main(){

    agentDQL* pAgent = new agentDQL({4,24,24,2},0.001,0.95,1.0);
    
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
            pAgent->test(500,5,true);
        }
        else if (option==3){
            pAgent->save_model();
        }
        else if (option==4){
            pAgent->load_model();
        }
    } while (option != 9);


    return 0;
}


