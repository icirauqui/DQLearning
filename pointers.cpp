#include <iostream>
#include <vector>

using namespace std;

vector<vector<int>* > vpv;



int main(){

    // Pointer: varible that holds the meory address of another variable
    //          Needs to be dereferenced with * operator to access the memory location it points to

    // Reference: variable that acts as an alias for another existing variable.
    //            Like a pointer, it's implemented by storing the address of an object.
    //            Can be thought of as a constant pointer (not a pointer to a constant value) with automatic indirection (the compiler applies the * operator)

    int n = 5;
    int *pn = &n;
    int &rn = n;

    int *pn2 = pn;

    cout << pn << " " << pn2 << " " << *pn << " " << *pn2 << endl;
    *pn = 3;    
    cout << pn << " " << pn2 << " " << *pn << " " << *pn2 << endl;

    cout << "n = " << n << "    &n = " << &n << "    rn = " << rn << "    pn = " << pn << "    *pn = " << *pn << endl;

    // A pointer can be reasigned
    // A reference cannot be reasigned, and must be assigned at initialization
    // A reference can be assigned to another reference, it just creates another alias



    // Pointer to a vector
    vector<int>* pv = new vector<int>(2);

    // Get values using 'pointer to' member
    cout << "pointer to = pv->at(0) = " << pv->at(0) << endl;

    // Get values using 'pointer to operator' member
    cout << "pointer to operator = pv->operator[](0) = " << pv->operator[](0) << endl;

    // Create a reference, access through it and delete the reference
    vector<int> &rv = *pv;
    rv[0] = 1;
    cout << "rv[0] = " << rv[0] << endl;
    delete &rv;

    //Pointer address
    cout << "pv = " << pv << "    &pv = " << &pv << endl;


    // Smart pointers
    // unique_ptr
    // shared_ptr
    // weak_ptr



    return 0;
}



