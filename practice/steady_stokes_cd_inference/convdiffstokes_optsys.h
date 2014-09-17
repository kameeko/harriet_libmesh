// DiffSystem framework files
#include "libmesh/fem_system.h"

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
    : FEMSystem(es, name_in, number_in){}

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
  
  //regularization parameter
  Real beta;

  // Returns the value of a forcing function at point p.  This value
  // depends on which application is being used.
  Point forcing(const Point& p);
};
