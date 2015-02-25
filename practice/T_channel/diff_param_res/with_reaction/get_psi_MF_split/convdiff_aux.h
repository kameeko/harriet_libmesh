#include "libmesh/fem_system.h"
#include "libmesh/getpot.h"

using namespace libMesh;

class ConvDiff_AuxSys : public FEMSystem
{
public:

  // Constructor
  ConvDiff_AuxSys(EquationSystems& es,
               const std::string& name_in,
               const unsigned int number_in)
    : FEMSystem(es, name_in, number_in){
    
    GetPot infile("convdiff_mprime.in");
		std::string find_velocity_here = infile("velocity_file","velsTtrim.txt");
		std::string find_data_here = infile("data_file","Measurements_top6.dat");
		qoi_option = infile("QoI_option",1);
    
    const unsigned int dim = this->get_mesh().mesh_dimension();
    
    if(FILE *fp=fopen(find_velocity_here.c_str(),"r")){
    	if(dim == 2){
				Real u, v, x, y;
				Real prevx = 1.e10;
				std::vector<Real> tempvecy;
				std::vector<NumberVectorValue> tempvecvel;
				int flag = 1;
				while(flag != -1){
					flag = fscanf(fp, "%lf %lf %lf %lf",&u,&v,&x,&y);
					if(flag != -1){
						if(x != prevx){
							x_pts.push_back(x);
							prevx = x;
							if(x_pts.size() > 1){
								y_pts.push_back(tempvecy);
								vel_field.push_back(tempvecvel);
							}
							tempvecy.clear(); 
							tempvecvel.clear();
							tempvecy.push_back(y); 
							tempvecvel.push_back(NumberVectorValue(u,v));
						}
						else{
							tempvecy.push_back(y); 
							tempvecvel.push_back(NumberVectorValue(u,v));
						}
					}
				}
				y_pts.push_back(tempvecy);
				vel_field.push_back(tempvecvel);
  		}
  	}
		if(FILE *fp=fopen(find_data_here.c_str(),"r")){
			if(dim == 2){
				Real x, y, value;
				int flag = 1;
				while(flag != -1){
					flag = fscanf(fp,"%lf %lf %lf",&x,&y,&value);
					if(flag != -1){
						datapts.push_back(Point(x,y));
						datavals.push_back(value);
					}
				}
				fclose(fp);
	  	}
	  	else if(dim == 1){
	  		Real x, value;
				int flag = 1;
				while(flag != -1){
					flag = fscanf(fp,"%lf %lf",&x,&value);
					if(flag != -1){
						datapts.push_back(Point(x));
						datavals.push_back(value);
					}
				}
				fclose(fp);
	  	}
	  }
	  accounted_for.assign(datavals.size(), this->get_mesh().n_elem()+100);
  }

  // System initialization
  virtual void init_data ();

  // Context initialization
  virtual void init_context(DiffContext &context);

  // Element residual and jacobian calculations
  // Time dependent parts
  virtual bool element_time_derivative (bool request_jacobian,
                                        DiffContext& context);
  

  // Indices for each variable;
  unsigned int aux_c_var, aux_zc_var, aux_fc_var;
  unsigned int aux_fpin_var;
  
  Real beta; //regularization parameter
  Real k; //diffusion coefficient
  Real R; //reaction coefficient
  
  //data-related stuff
  std::vector<Point> datapts; 
  std::vector<Real> datavals;
  
  //velocity field
	std::vector<Real> x_pts;
	std::vector<std::vector<Real> > y_pts;
	std::vector<std::vector<NumberVectorValue> > vel_field;
	
	int scalar_subdomain_id, field_subdomain_id;
	
	//avoid assigning data point to two elements in on their boundary
	std::vector<int> accounted_for;
  
  //options for QoI location and nature
  int qoi_option;
  
  double screw_mag, screw_grad; //knobs for how hard to enfore pinning to constant
};
