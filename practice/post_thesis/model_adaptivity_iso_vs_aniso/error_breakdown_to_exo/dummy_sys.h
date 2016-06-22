// DiffSystem framework files
#include "libmesh/fem_system.h"
#include "libmesh/getpot.h"
#include "libmesh/fem_context.h"
#include "libmesh/equation_systems.h"
#include "libmesh/boundary_info.h"
#include "libmesh/dirichlet_boundaries.h"
#include "libmesh/dof_map.h"
#include "libmesh/fe_base.h"
#include "libmesh/fe_interface.h"
#include "libmesh/mesh.h"
#include "libmesh/quadrature.h"
#include "libmesh/string_to_enum.h"
#include "libmesh/zero_function.h"

using namespace libMesh;

// FEMSystem, TimeSolver and  NewtonSolver will handle most tasks,
// but we must specify element residuals
class DummySys : public FEMSystem
{
public:

  // Constructor
  DummySys(EquationSystems& es, const std::string& name_in, const unsigned int number_in):
    FEMSystem(es, name_in, number_in){}

  // System initialization
  void init_data(){
    const unsigned int dim = this->get_mesh().mesh_dimension();

    GetPot infile("contamTrans.in");
    unsigned int poly_order = infile("poly_order",1);
    std::string fefamily = infile("fe_family", std::string("LAGRANGE"));

    c_var = this->add_variable("local_error", static_cast<Order>(poly_order), Utility::string_to_enum<FEFamily>(fefamily));

	  // Do the parent's initialization after variables and boundary constraints are defined
	  FEMSystem::init_data();
  }

  // Context initialization
  void init_context(DiffContext & context){
    FEMContext &ctxt = cast_ref<FEMContext&>(context);

    FEBase* c_elem_fe;
    FEBase* c_side_fe;

    ctxt.get_element_fe(c_var, c_elem_fe);

	  c_elem_fe->get_JxW();
	  c_elem_fe->get_phi();
	  c_elem_fe->get_xyz();
  }


  // Element residual and jacobian calculations
  // Time dependent parts
  bool element_time_derivative(bool request_jacobian, DiffContext & context)
  {
    const unsigned int dim = this->get_mesh().mesh_dimension();

    FEMContext &ctxt = cast_ref<FEMContext&>(context);

    FEBase* c_elem_fe = NULL;
    ctxt.get_element_fe( c_var, c_elem_fe );

    // Element Jacobian * quadrature weights for interior integration
    const std::vector<Real> &JxW = c_elem_fe->get_JxW();

    // Physical location of the quadrature points
    const std::vector<Point>& qpoint = c_elem_fe->get_xyz();

    // The number of local degrees of freedom in each variable
    const unsigned int n_c_dofs = ctxt.get_dof_indices( c_var ).size();

    // The subvectors and submatrices we need to fill:
    DenseSubMatrix<Number> &J = ctxt.get_elem_jacobian(c_var, c_var);
    DenseSubVector<Number> &R = ctxt.get_elem_residual(c_var);

    // Now we will build the element Jacobian and residual.
    // Constructing the residual requires the solution and its
    // gradient from the previous timestep.  This must be
    // calculated at each quadrature point by summing the
    // solution degree-of-freedom values by the appropriate
    // weight functions.
    unsigned int n_qpoints = ctxt.get_element_qrule().n_points();

    for (unsigned int qp=0; qp != n_qpoints; qp++)
    {

      // First, an i-loop over the  degrees of freedom.
      for (unsigned int i=0; i != n_c_dofs; i++)
      {
        // The residual
        R(i) += 0.;
	    
        if (request_jacobian && ctxt.get_elem_solution_derivative())
        {
          for (unsigned int j=0; j != n_c_dofs; j++)
          {
            J(i,j) += 0.;
          } // end of the inner dof (j) loop
        } // end - if (request_jacobian && context.get_elem_solution_derivative())

      } // end of the outer dof (i) loop
    } // end of the quadrature point (qp) loop

    return request_jacobian;
  }

  // Indices for each variable;
  unsigned int c_var;
};
