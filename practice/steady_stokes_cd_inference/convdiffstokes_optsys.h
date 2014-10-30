// DiffSystem framework files
#include "libmesh/fem_system.h"

using namespace libMesh;

//this one is for the optimality system, not just the forward

// FEMSystem, TimeSolver and  NewtonSolver will handle most tasks,
// but we must specify element residuals
class StokesConvDiffSys : public FEMSystem
{
public:

  // Constructor
  StokesConvDiffSys(EquationSystems& es,
               const std::string& name_in,
               const unsigned int number_in)
    : FEMSystem(es, name_in, number_in){
    
    if(FILE *fp=fopen("Measurements.dat","r")){
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

  // Indices for each variable;
  unsigned int p_var, u_var, v_var, c_var, zp_var, zu_var, zv_var, zc_var, fc_var;
  
  Real beta; //regularization parameter
  Real regtype; // 0 = squared L2 norm; 1 = penalize first derivative
  
  //data-related stuff
  std::vector<Point> datapts; 
  std::vector<Real> datavals;

  // Returns the value of a forcing function at point p.  This value
  // depends on which application is being used.
  Point forcing(const Point& p);
};
