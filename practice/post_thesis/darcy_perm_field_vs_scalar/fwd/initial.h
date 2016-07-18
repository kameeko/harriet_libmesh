#include "libmesh/parameters.h"
#include "libmesh/point.h"
#include "libmesh/vector_value.h"

using namespace libMesh;

void read_initial_parameters(){}
void finish_initialization(){}

Number initial_value(const Point& p, const Parameters& parameters, 
    const std::string& sys_name, const std::string& unknown_name){
  return 5.0; //ppb
}

Gradient initial_grad(const Point& p, const Parameters& parameters, 
    const std::string& sys_name, const std::string& unknown_name){
  return Gradient(0.0, 0.0, 0.0);
}
