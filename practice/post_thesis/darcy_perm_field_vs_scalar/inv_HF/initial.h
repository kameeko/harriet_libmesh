#include "libmesh/parameters.h"
#include "libmesh/point.h"
#include "libmesh/vector_value.h"

using namespace libMesh;

void read_initial_parameters(){}
void finish_initialization(){}

Number initial_value(const Point& p, const Parameters& parameters, 
    const std::string& sys_name, const std::string& unknown_name){
    
  Number init_val = 0.;   
  if(unknown_name == "k"){
    //init_val = 2.72e-10; //avg_perm; m^2
    init_val = 275.8; //avg_perm; darcies
  }  
  return init_val;
}

Gradient initial_grad(const Point& p, const Parameters& parameters, 
    const std::string& sys_name, const std::string& unknown_name){
  return Gradient(0.0, 0.0, 0.0);
}
