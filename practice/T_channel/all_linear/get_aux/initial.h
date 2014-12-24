#include "libmesh/parameters.h"
#include "libmesh/point.h"
#include "libmesh/vector_value.h"

//FOR 1D DEBUG
//AUX AND PRIMARY FLIPPED BECAUSE VARIABLE NAMES MISLEADING IN THIS ONE

using namespace libMesh;

void read_initial_parameters()
{
std::cout << "\n\n For 20-element coarse version only! \n\n";
}

void finish_initialization()
{
}



// Initial conditions
Number initial_value(	const Point &p,
											const Parameters& parameters,
											const std::string& sys_name,
											const std::string& unknown_name)
{

	Number returnme;
  if(FILE *fp=fopen(std::string("split_psi_hf1.txt").c_str(),"r")){
  	Real c, zc, fc1, fc2, fc3, fc4, fc5, auxc, auxzc, auxfc1, auxfc2, auxfc3, auxfc4, auxfc5;
  	Real x;
  	Real leftval, rightval, leftx, rightx;
  	int flag = 1;
		while(flag != -1){
			flag = fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
							&x, &auxc, &auxzc, &auxfc1, &auxfc2, &auxfc3, &auxfc4, &auxfc5, &c, &zc, &fc1, &fc2, &fc3, &fc4, &fc5);
			if(flag != -1){
				if(unknown_name == "fc1"){returnme = fc1; flag = -1;}
				else if(unknown_name == "fc2"){returnme = fc2; flag = -1;}
				else if(unknown_name == "fc3"){returnme = fc3; flag = -1;}
				else if(unknown_name == "fc4"){returnme = fc4; flag = -1;}
				else if(unknown_name == "fc5"){returnme = fc5; flag = -1;}
				else if(unknown_name == "aux_fc1"){returnme = auxfc1; flag = -1;}
				else if(unknown_name == "aux_fc2"){returnme = auxfc2; flag = -1;}
				else if(unknown_name == "aux_fc3"){returnme = auxfc3; flag = -1;}
				else if(unknown_name == "aux_fc4"){returnme = auxfc4; flag = -1;}
				else if(unknown_name == "aux_fc5"){returnme = auxfc5; flag = -1;}
			
				else if(p(0) >= x && p(0) < x+0.05){
					if(unknown_name == "c") {leftval = c; leftx = x;}
					else if(unknown_name == "zc") {leftval = zc; leftx = x;}
					else if(unknown_name == "aux_c") {leftval = auxc; leftx = x;}
					else if(unknown_name == "aux_zc") {leftval = auxzc; leftx = x;}
				}
				else if(p(0) < x && p(0) >= x-0.05){
					if(unknown_name == "c"){rightval = c; rightx = x; flag = -1;}
					else if(unknown_name == "zc"){rightval = zc; rightx = x; flag = -1;}
					else if(unknown_name == "aux_c"){rightval = auxc; rightx = x; flag = -1;}
					else if(unknown_name == "aux_zc"){rightval = auxzc; rightx = x; flag = -1;}
				}
			}
		}
		if(unknown_name == "c" || unknown_name == "zc" || unknown_name == "aux_c" || unknown_name == "aux_zc"){
			returnme = leftval + (p(0)-leftx)*(rightval-leftval)/(rightx-leftx);
		}
  }

  return returnme;
}



Gradient initial_grad(const Point &p,
											const Parameters& parameters,
											const std::string& sys_name,
											const std::string& unknown_name)
{
  Real returnme;
  if(unknown_name == "c" || unknown_name == "zc" || unknown_name == "aux_c" || unknown_name == "aux_zc"){
  	if(FILE *fp=fopen(std::string("split_psi_hf1.txt").c_str(),"r")){
			Real c, zc, fc1, fc2, fc3, fc4, fc5, auxc, auxzc, auxfc1, auxfc2, auxfc3, auxfc4, auxfc5;
			Real x;
			Real leftval, rightval, leftx, rightx;
			int flag = 1;
			while(flag != -1){
				flag = fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
								&x, &c, &zc, &fc1, &fc2, &fc3, &fc4, &fc5, &auxc, &auxzc, &auxfc1, &auxfc2, &auxfc3, &auxfc4, &auxfc5);
				if(flag != -1){
					if(p(0) >= x && p(0) < x+0.05){
						if(unknown_name == "c") {leftval = c; leftx = x;}
						else if(unknown_name == "zc") {leftval = zc; leftx = x;}
						else if(unknown_name == "aux_c") {leftval = auxc; leftx = x;}
						else if(unknown_name == "aux_zc") {leftval = auxzc; leftx = x;}
					}
					else if(p(0) < x && p(0) >= x-0.05){
						if(unknown_name == "c"){rightval = c; rightx = x; flag = -1;}
						else if(unknown_name == "zc"){rightval = zc; rightx = x; flag = -1;}
						else if(unknown_name == "aux_c"){rightval = auxc; rightx = x; flag = -1;}
						else if(unknown_name == "aux_zc"){rightval = auxzc; rightx = x; flag = -1;}
					}
				}
			}
			returnme = (rightval-leftval)/(rightx-leftx);
		}
  }
  else
  	std::cout << "\n...why does it want the gradient?\n";
  
  return returnme;
}

