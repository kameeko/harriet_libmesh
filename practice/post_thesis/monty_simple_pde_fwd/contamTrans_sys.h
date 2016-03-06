// DiffSystem framework files
#include "libmesh/fem_system.h"

using namespace libMesh;

// FEMSystem, TimeSolver and  NewtonSolver will handle most tasks,
// but we must specify element residuals
class ContamTransSys : public FEMSystem
{
public:

  // Constructor
  ContamTransSys(EquationSystems& es, const std::string& name_in, const unsigned int number_in):
    FEMSystem(es, name_in, number_in){}

  // System initialization
  virtual void init_data ();

  // Context initialization
  virtual void init_context(DiffContext &context);

  // Element residual and jacobian calculations
  // Time dependent parts
  virtual bool element_time_derivative (bool request_jacobian,
                                        DiffContext& context);

  //boundary residual and jacobian calculations
  virtual bool side_time_derivative (bool request_jacobian,
                                        DiffContext& context);

  // Postprocessed output
  virtual void postprocess ();

  // Indices for each variable;
  unsigned int c_var;

  // Returns the value of a forcing function at point p.  This value
  // depends on which application is being used.
  Point forcing(const Point& p);

  Real vx; //west-east velocity (along x-axis, for now); m/s
  Real react_rate; //reaction rate; 1/s
  Real porosity; //porosity
  Real bsource; // ppb (concentration of influx at west boundary)
  Real source_rate; // kg/s
	Real source_conc; // ppb
	Real water_density; // kg/m^3
  NumberTensorValue dispTens; //dispersion tensor; m^2/s
  std::vector<Real> xlim; //boundaries of interior source; m
  std::vector<Real> ylim; //boundaries of interior source; m
  Real source_zmax; //top of domain; m
  Real source_dz; //interior source thickness; m
  Real source_vol; //interior source volume; m^3
  
  bool useSUPG; //whether to use SUPG...
  Real ds; //decay scaling parameter for smoothing edges of box source; larger for faster decay
  int stab_opt; //stabilization options
  
  bool diri_dbg; //whether to use Dirichlet boundary conditions (for debugging)
  bool smooth_box; //whether to use box-source with smoothed edges (exponential decay)
};
