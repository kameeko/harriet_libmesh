#include "initial.h"

using namespace libMesh;

void read_initial_parameters()
{
}

void finish_initialization()
{
}



// Initial conditions
Number initial_value(const Point& p,
                     const Parameters&,
                     const std::string&,
                     const std::string& unknown_name)
{
 Real x = p(0), y = p(1);
 
 Real val = 0.0;

 if(unknown_name == "K")
   {
     val = 1.0;     
   }
 else if(unknown_name == "p")
   {
     val = 0.0;     
   }
 else if(unknown_name == "z")
   {
     val = 0.0;     
   } 
 else
   {
     std::cout<<"Unknown variable: "<<unknown_name<<std::endl;
     libmesh_error();
   }
 
 return val;
}

Gradient initial_grad(const Point& p,
                      const Parameters&,
                      const std::string&,
                      const std::string& unknown_name)
{
  Real x = p(0), y = p(1);

  Gradient grad_val(0., 0.);

  //if(unknown_name == "h")
  //{
  // grad_val(0) = M_PI*cos(M_PI * x) * sin(M_PI * y);
  // grad_val(1) = M_PI*sin(M_PI * x) * cos(M_PI * y) ;
  //}

  return grad_val;
}
