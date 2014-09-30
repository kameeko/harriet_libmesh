#include "libmesh/enum_fe_family.h"
#include "libmesh/fem_system.h"
#include "libmesh/parameter_vector.h"
#include "libmesh/qoi_set.h"
#include "libmesh/system.h"

using namespace libMesh;

// FEMSystem, TimeSolver and  NewtonSolver will handle most tasks,
// but we must specify element residuals
class NavStokesConvDiffSys : public FEMSystem
{
public:

  // Constructor
  NavStokesConvDiffSys(EquationSystems& es,
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
    return params[parameter_index];
  }

  ParameterVector &get_parameter_vector()
  {
    param_vector.resize(params.size());
    for(unsigned int i = 0; i != params.size(); ++i)
      {
        param_vector[i] = &params[i];
      }

    return param_vector;
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
  unsigned int p_var, u_var, v_var, c_var, zp_var, zu_var, zv_var, zc_var, fc_var,
  						aux_u_var, aux_v_var, aux_p_var, aux_c_var,
  						aux_zu_var, aux_zv_var, aux_zp_var, aux_zc_var, aux_fc_var;
  
  //density and viscosity
  Real rho, mu;
  
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
  std::vector<Number> params;

  // The ParameterVector object that will contain pointers to
  // the system parameters
  ParameterVector param_vector;
};
