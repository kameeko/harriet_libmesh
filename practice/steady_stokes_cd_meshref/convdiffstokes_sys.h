#include "libmesh/enum_fe_family.h"
#include "libmesh/fem_system.h"
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
    _analytic_jacobians(true) { qoi.resize(1); }
  
  bool & analytic_jacobians() { return _analytic_jacobians; }
  
  virtual void postprocess(void);

  Number &get_QoI_value(std::string type, unsigned int QoI_index){
      return computed_QoI[QoI_index]; //no exact QoI available
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
  unsigned int p_var, u_var, v_var, c_var;

  // Returns the value of a forcing function at point p.  This value
  // depends on which application is being used.
  Point forcing(const Point& p);
  
  virtual void element_qoi_derivative(DiffContext &context, const QoISet & qois);
  
  //to hold computed QoI
  Number computed_QoI[1];
  
  // Calculate Jacobians analytically or not?
  bool _analytic_jacobians;
};
