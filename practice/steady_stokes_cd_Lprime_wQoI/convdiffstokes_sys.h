#include "libmesh/enum_fe_family.h"
#include "libmesh/fem_system.h"
#include "libmesh/parameter_vector.h"
#include "libmesh/qoi_set.h"
#include "libmesh/system.h"

using namespace libMesh;

// FEMSystem, TimeSolver and  NewtonSolver will handle most tasks,
// but we must specify element residuals
class StokesConvDiffSys : public FEMSystem
{
public:

  // Constructor
  StokesConvDiffSys(EquationSystems& es,
               const std::string& name_in,
               const unsigned int number_in)
    : FEMSystem(es, name_in, number_in),
    _analytic_jacobians(true) {
    
    qoi.resize(1); //only one QoI
    
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
  } //end constructor
  
  bool & analytic_jacobians() { return _analytic_jacobians; }
  
  virtual void postprocess(void);

  Number &get_QoI_value(std::string type, unsigned int QoI_index){
      return computed_QoI[QoI_index]; //no exact QoI available
  }
  
  Number &get_parameter_value(unsigned int parameter_index)
  {
    return parameters[parameter_index];
  }

  ParameterVector &get_parameter_vector()
  {
    parameter_vector.resize(parameters.size());
    for(unsigned int i = 0; i != parameters.size(); ++i)
      {
        parameter_vector[i] = &parameters[i];
      }

    return parameter_vector;
  }
  
protected:

  // System initialization
  virtual void init_data ();

  // Context initialization
  virtual void init_context(DiffContext &context);

  // Element residual and jacobian calculations
  // Time dependent parts
  virtual bool element_time_derivative (bool request_jacobian,
                                        DiffContext& context);

  //to calculate QoI
  virtual void element_postprocess(DiffContext &context);

  // Indices for each variable;
  unsigned int p_var, u_var, v_var, c_var, zp_var, zu_var, zv_var, zc_var, fc_var;
  
  //Peclet number; parameter to get sensitivity for
  Real Peclet;
  
  virtual void element_qoi_derivative(DiffContext &context, const QoISet & qois);
  
  //to hold computed QoI
  Number computed_QoI[1];
  
  // Calculate Jacobians analytically or not?
  bool _analytic_jacobians;
  
  Real beta; //regularization parameter
  Real regtype; // 0 = squared L2 norm; 1 = penalize first derivative
  
  //data-related stuff
  std::vector<Point> datapts; 
  std::vector<Real> datavals;
  
  // Parameters associated with the system
  std::vector<Number> parameters;

  // The ParameterVector object that will contain pointers to
  // the system parameters
  ParameterVector parameter_vector;
};
