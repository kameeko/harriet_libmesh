#include "libmesh/fem_system.h"
#include "libmesh/getpot.h"

using namespace libMesh;

//this one is for the optimality system, not just the forward

// FEMSystem, TimeSolver and  NewtonSolver will handle most tasks,
// but we must specify element residuals
class Diff_ConvDiff_Sys : public FEMSystem
{
public:

  // Constructor
  Diff_ConvDiff_Sys(EquationSystems& es,
               const std::string& name_in,
               const unsigned int number_in)
    : FEMSystem(es, name_in, number_in){
    
    GetPot infile("diff_convdiff_inv.in");
		std::string find_velocity_here = infile("velocity_file","velsTtrim.txt");
		std::string write_data_here = infile("data_file","Measurements.dat");
		
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
		if(FILE *fp=fopen(write_data_here.c_str(),"r")){
			Real x, y, value;
			int flag = 1;
			while(flag != -1){
				flag = fscanf(fp,"%lf %lf %lf",&x,&y,&value);
				if(flag != -1){
					datapts.push_back(Point(x,y));
					datavals.push_back(0.0);
				}
			}
			fclose(fp);
	  }
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
 
  Point forcing(const Point& p);
  

  // Indices for each variable;
  unsigned int c_var;
  
  Real k; //diffusion coefficient
  Real Rcoeff; //reaction coefficient
  
  //data-related stuff
  std::vector<Point> datapts; 
  std::vector<Real> datavals;
  
  //velocity field
	std::vector<Real> x_pts;
	std::vector<std::vector<Real> > y_pts;
	std::vector<std::vector<NumberVectorValue> > vel_field;

};