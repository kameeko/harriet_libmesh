#include "libmesh/fem_system.h"
#include "libmesh/getpot.h"

using namespace libMesh;

//this one is for the optimality system, not just the forward

// FEMSystem, TimeSolver and  NewtonSolver will handle most tasks,
// but we must specify element residuals
class Diff_ConvDiff_InvSys : public FEMSystem
{
public:

  // Constructor
  Diff_ConvDiff_InvSys(EquationSystems& es,
               const std::string& name_in,
               const unsigned int number_in)
    : FEMSystem(es, name_in, number_in){
    
    GetPot infile("diff_convdiff_inv.in");
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

  // Postprocessed output
  virtual void postprocess ();

  //to calculate QoI
  virtual void element_postprocess(DiffContext &context);
  
  //return QoI
  Number &get_QoI_value(std::string type, unsigned int QoI_index){
      return computed_QoI[QoI_index]; //no exact QoI available
  }
  
  //calculate forcing function corresponding to basis coefficients; 1D debugging
  Real f_from_coeff(Real fc1, Real fc2, Real fc3, Real fc4, Real fc5, Real x);
  
  

  // Indices for each variable;
  unsigned int c_var, zc_var, fc_var;
  unsigned int fc1_var, fc2_var, fc3_var, fc4_var, fc5_var; //for 1D debugging
  
  Real beta; //regularization parameter
  Real k; //diffusion coefficient
  
  //data-related stuff
  std::vector<Point> datapts; 
  std::vector<Real> datavals;
  
  //velocity field
	std::vector<Real> x_pts;
	std::vector<std::vector<Real> > y_pts;
	std::vector<std::vector<NumberVectorValue> > vel_field;
	
	int diff_subdomain_id, convdiff_subdomain_id;
	
	//avoid assigning data point to two elements in on their boundary
	std::vector<int> accounted_for;
	
  //to hold computed QoI
  Number computed_QoI[1];
  
  //options for QoI location and nature
  int qoi_option;
};
