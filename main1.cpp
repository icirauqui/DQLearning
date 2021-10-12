#include <iostream>

#include "agentDQL1.cpp"

int main(){

    agentDQL* pAgent = new agentDQL({4,24,24,2},0.001,0.95,1.0);

    int layers = 10;

    for (int i=layers-1; i>=0; i--)
        std::cout << i << " ";
    std::cout << std::endl;
    
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


    return 0;





}
