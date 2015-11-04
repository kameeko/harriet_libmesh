#include "libmesh/getpot.h"

#include "contamTrans_sys.h"

#include "libmesh/boundary_info.h"
#include "libmesh/dirichlet_boundaries.h"
#include "libmesh/dof_map.h"
#include "libmesh/fe_base.h"
#include "libmesh/fe_interface.h"
#include "libmesh/fem_context.h"
#include "libmesh/mesh.h"
#include "libmesh/quadrature.h"
#include "libmesh/string_to_enum.h"
#include "libmesh/zero_function.h"

// Bring in everything from the libMesh namespace
using namespace libMesh;

//system initialization
void ContamTransSys::init_data(){
  const unsigned int dim = this->get_mesh().mesh_dimension();

  GetPot infile("contamTrans.in");
  unsigned int poly_order = infile("poly_order",1);
  std::string fefamily = infile("fe_family", std::string("LAGRANGE"));

  c_var = this->add_variable("c", static_cast<Order>(poly_order), Utility::string_to_enum<FEFamily>(fefamily));

  //indicate variables that change in time
  this->time_evolving(c_var);

  // Useful debugging options'
	// Set verify_analytic_jacobians to positive to use
	this->verify_analytic_jacobians = infile("verify_analytic_jacobians", 0.);
	this->print_jacobians = infile("print_jacobians", false);
	this->print_element_jacobians = infile("print_element_jacobians", false);

	//set Dirichlet boundary conditions (none in this case)

	//set parameters
	vx = infile("vx",2.415e-5);
	react_rate = infile("reaction_rate",0.0);
	porosity = infile("porosity",0.1);
	double dlong = infile("dispersivity_longitudinal",60.0);
	double dtransh = infile("dispersivity_transverse_horizontal",6.0);
	double dtransv;
	if(dim == 2)
	  dtransv = 0.0;
	else if(dim == 3)
	  dtransv = infile("dispersivity_transverse_vertical",0.6);

	//compute dispersion tensor (assuming for now that velocity purely in x direction)
	dispTens = NumberTensorValue(vx*dlong, 0.0, 0.0,
	                            0.0, vx*dtransh, 0.0,
	                            0.0, 0.0, vx*dtransv);

	// Do the parent's initialization after variables and boundary constraints are defined
	FEMSystem::init_data();
}

//context initialization
void ContamTransSys::init_context(DiffContext & context){
  FEMContext &ctxt = cast_ref<FEMContext&>(context);

  FEBase* c_elem_fe;
  FEBase* c_side_fe;

  ctxt.get_element_fe(c_var, c_elem_fe);
	ctxt.get_side_fe(c_var, c_side_fe );

	c_elem_fe->get_JxW();
	c_elem_fe->get_phi();
	c_elem_fe->get_dphi();
	c_elem_fe->get_xyz();

	c_side_fe->get_JxW();
	c_side_fe->get_phi();
	c_side_fe->get_dphi();
	c_side_fe->get_xyz();
}

//element residual and jacobian calculations
bool ContamTransSys::element_time_derivative(bool request_jacobian, DiffContext & context)
{
  const unsigned int dim = this->get_mesh().mesh_dimension();

  FEMContext &ctxt = cast_ref<FEMContext&>(context);

  FEBase* c_elem_fe = NULL;
  ctxt.get_element_fe( c_var, c_elem_fe );

  // Element Jacobian * quadrature weights for interior integration
  const std::vector<Real> &JxW = c_elem_fe->get_JxW();

  const std::vector<std::vector<Real> >& phi = c_elem_fe->get_phi();
  const std::vector<std::vector<RealGradient> >& dphi = c_elem_fe->get_dphi();

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
    Number c = ctxt.interior_value(c_var, qp);
    Gradient grad_c = ctxt.interior_gradient(c_var, qp);

    Point f = this->forcing(qpoint[qp]);
    Number fc = f(0);

    //velocity vector
    NumberVectorValue U(vx,0.0);
    if(dim > 2)
      U(2) = 0.0;

    // First, an i-loop over the  degrees of freedom.
    for (unsigned int i=0; i != n_c_dofs; i++)
    {
      // The residual
      R(i) += JxW[qp]*(-(dispTens*(porosity*grad_c))*dphi[i][qp] // Dispersion Term
		       - (U*grad_c)*phi[i][qp] // Convection Term
		       - (react_rate*(porosity*c))*phi[i][qp] // Reaction Term
		       + fc*phi[i][qp]); // Source term

      if (request_jacobian)
      {
	for (unsigned int j=0; j != n_c_dofs; j++)
	{
	  J(i,j) += JxW[qp]*(-dispTens*porosity*dphi[j][qp]*dphi[i][qp] // Dispersion
			     - U*dphi[j][qp]*phi[i][qp] // Convection
			     + react_rate*phi[j][qp]*phi[i][qp]); // Reaction Term

	} // end of the inner dof (j) loop
      } // end - if (compute_jacobian && context.get_elem_solution_derivative())

    } // end of the outer dof (i) loop
  } // end of the quadrature point (qp) loop

  return request_jacobian;
}

//for non-Dirichlet boundary conditions and the bit from diffusion term
bool ContamTransSys::side_constraint(bool request_jacobian, DiffContext & context)
{
  const unsigned int dim = this->get_mesh().mesh_dimension();

  FEMContext &ctxt = cast_ref<FEMContext&>(context);

  // First we get some references to cell-specific data that
  // will be used to assemble the linear system.
  FEBase* side_fe = NULL;
  ctxt.get_side_fe(c_var, side_fe );

  // Element Jacobian * quadrature weights for interior integration
  const std::vector<Real> &JxW = side_fe->get_JxW();

  // Side basis functions
  const std::vector<std::vector<Real> > &phi = side_fe->get_phi();
  const std::vector<std::vector<RealGradient> > &dphi = side_fe->get_dphi();

  // Side Quadrature points
  const std::vector<Point > &qside_point = side_fe->get_xyz();

  //normal vector
  const std::vector<Point> &face_normals = side_fe->get_normals();

  // The number of local degrees of freedom in each variable
  const unsigned int n_c_dofs = ctxt.get_dof_indices(c_var).size();

  // The subvectors and submatrices we need to fill:
  DenseSubMatrix<Number> &J = ctxt.get_elem_jacobian(c_var, c_var);
	DenseSubVector<Number> &R  = ctxt.get_elem_residual(c_var);

  unsigned int n_qpoints = ctxt.get_side_qrule().n_points();

  bool isWest = ctxt.has_side_boundary_id(4); //guessing this one is west...?

  for (unsigned int qp=0; qp != n_qpoints; qp++)
  {
    Number c = ctxt.side_value(c_var, qp);
    Gradient grad_c = ctxt.side_gradient(c_var, qp);

    //velocity vector
    NumberVectorValue U(vx,0.0);

    if(dim > 2)
      U(2) = 0.0;

    for (unsigned int i=0; i != n_c_dofs; i++)
    {
      //bit from changing order on dispersion term
      R(i) += JxW[qp]*((dispTens*(porosity*grad_c))*face_normals[qp])*phi[i][qp]; // Dispersion term

      //flux boundary conditions
      if(isWest)
      {
	//west boundary
        double bsource = -5.0; //ppb (doesn't seem to be the right units for flux though?)
        R(i) += JxW[qp]*(U*face_normals[qp]*c - (dispTens*grad_c)*face_normals[qp] - bsource)*phi[i][qp];
        std::cout << qside_point[qp](0) << " " << qside_point[qp](1) << " " << qside_point[qp](2) << std::endl; //DEBUG
      }
      else //"mass flow out" boundary condition?
        R(i) += JxW[qp]*(dispTens*grad_c)*face_normals[qp]*phi[i][qp];

      if(request_jacobian)
      {
        for (unsigned int j=0; j != n_c_dofs; j++)
	{
          J(i,j) += JxW[qp]*((dispTens*dphi[j][qp])*face_normals[qp])*phi[i][qp];
          if(isWest)
            J(i,j) += JxW[qp]*(U*face_normals[qp]*phi[j][qp] - (dispTens*dphi[j][qp])*face_normals[qp])*phi[i][qp];
          else
            J(i,j) += JxW[qp]*((dispTens*dphi[j][qp])*face_normals[qp])*phi[i][qp];
        }
      }
    }
  }

  return request_jacobian;
}

//generate data
void ContamTransSys::postprocess(){
  std::ostringstream file_name;
  file_name << "Measurements" << time << ".dat";
  std::ofstream output(file_name.str());
  //write out data
  output.close();
}

//source term at point pt
Point ContamTransSys::forcing(const Point& pt){
	Point f;

	std::vector<double> xlim{498316.0, 498716.0};
  std::vector<double> ylim{538742.0, 539522.0};

  double zmax = 100.0;
  double ztol = 2.0; //a really thin box...

	if(pt(0) >= xlim[0] && pt(0) <= xlim[1] && pt(1) >= ylim[0] && pt(1) <= ylim[1] && abs(pt(2)-zmax) <= ztol)
	  f(0) = 1000; //ppb
	else
	  f(0) = 0.0;

	return f;
}
