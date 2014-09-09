// DiffSystem framework files
#include "libmesh/fem_system.h"
#include "libmesh/point.h"

using namespace libMesh;

// The PorousHF system class.
// FEMSystem, TimeSolver and  NewtonSolver will handle most tasks,
// but we must specify element residuals
class PorousHFSystem : public FEMSystem
{
public:
  // Constructor
  PorousHFSystem(EquationSystems& es,
               const std::string& name,
               const unsigned int number)
    : FEMSystem(es, name, number), _analytic_jacobians(false) {}

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
      
  // Regularization parameter
  Real alpha;
  
  // Indices for each variable;
  unsigned int K_var, p_var, z_var;
    
  // Variables to hold the computed QoIs
  Number computed_QoI[2];
    
  // Calculate Jacobians analytically or not?
  bool _analytic_jacobians;
    
};
