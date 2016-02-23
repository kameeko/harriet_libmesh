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
	vx = infile("vx", 2.415e-5); // m/s
	react_rate = infile("reaction_rate", 0.0); // 1/s
	porosity = infile("porosity", 0.1); // (unitless)
	bsource = infile("bsource", -5.0); // ppb
	source_rate = infile("source_rate", 5.0); // kg/s
	source_conc = infile("source_conc", 1000.0); // ppb
	source_dz = infile("source_thickness", 1.0); // m
	
	xlim.push_back(498316.0); xlim.push_back(498716.0); // m
  ylim.push_back(538742.0); ylim.push_back(539522.0); // m
	source_vol = (xlim[1] - xlim[0])*(ylim[1] - ylim[0])*source_dz;
	water_density = 1.0; // kg/m^3
	source_zmax = 100.0; // m

	//compute dispersion tensor (assuming for now that velocity purely in x direction)
	double dlong = infile("dispersivity_longitudinal",60.0);
	double dtransh = infile("dispersivity_transverse_horizontal",6.0);
	double dtransv = infile("dispersivity_transverse_vertical",0.6);
	dispTens = NumberTensorValue(vx*dlong, 0.0, 0.0,
	                            0.0, vx*dtransh, 0.0,
	                            0.0, 0.0, vx*dtransv);

  useSUPG = infile("use_SUPG",false);
  ds = infile("source_decay",100.0);

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
    NumberVectorValue U(vx, 0.0, 0.0);
    
    //SUPG
    double tau = 0.0;
    if(useSUPG){ //assuming isotropic dispersion for now
      //version 1, copied from another code
      double C1 = 4.0;
      double C2 = 2.0;
      double k = dispTens(0,0);
      if(dispTens(0,0) != dispTens(1,1) || dispTens(1,1) != dispTens(2,2))
        std::cout << "SUPG currently assumed isotropic dispersion..." << std::endl;
      Real h = ctxt.get_elem().hmax();
      tau = 1./((C1*k)/(h*h) + (C2*sqrt(U*U)/h));
      
      //version 2, from http://ta.twi.tudelft.nl/TWA_Reports/06/06-03.pdf
      /*double k = dispTens(0,0);
      if(dispTens(0,0) != dispTens(1,1) || dispTens(1,1) != dispTens(2,2))
        std::cout << "SUPG currently assumed isotropic dispersion..." << std::endl;
      Real h = ctxt.get_elem().hmax();
      Real Pe = sqrt(U*U)*h/(2.*k); //element Peclet number
      tau = (h/(2.*sqrt(U*U)))*(1./tanh(Pe) + 1./Pe);*/
    }

    // First, an i-loop over the  degrees of freedom.
    for (unsigned int i=0; i != n_c_dofs; i++)
    {
      // The residual
      R(i) += JxW[qp]*
           (-(dispTens*(porosity*grad_c))*dphi[i][qp] // Dispersion Term
		       - (U*grad_c)*phi[i][qp] // Convection Term
		       - (react_rate*(porosity*c))*phi[i][qp] // Reaction Term
		       + fc*phi[i][qp]); // Source term
		  if(useSUPG)
		    R(i) += JxW[qp]*((tau*U*dphi[i][qp])*
		           (- (U*grad_c) // Convection Term
		           - (react_rate*(porosity*c)) // Reaction Term
		           + fc)); // Source term
      if (request_jacobian && ctxt.get_elem_solution_derivative())
      {
	      for (unsigned int j=0; j != n_c_dofs; j++)
	      {
	        J(i,j) += JxW[qp]*
	               ((-dispTens*(porosity*dphi[j][qp]))*dphi[i][qp] // Dispersion
			           - (U*dphi[j][qp])*phi[i][qp] // Convection
			           - (react_rate*(porosity*phi[j][qp]))*phi[i][qp]); // Reaction Term
			    if(useSUPG)
			      J(i,j) += JxW[qp]*((tau*U*dphi[i][qp])* 
		                 (- (U*dphi[j][qp]) // Convection Term
		                 - (react_rate*(porosity*phi[j][qp])))); // Reaction Term
	      } // end of the inner dof (j) loop
      } // end - if (request_jacobian && context.get_elem_solution_derivative())

    } // end of the outer dof (i) loop
  } // end of the quadrature point (qp) loop

  return request_jacobian;
}

//for non-Dirichlet boundary conditions and the bit from diffusion term
bool ContamTransSys::side_time_derivative(bool request_jacobian, DiffContext & context)
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

  // Side Quadrature points
  const std::vector<Point > &qside_point = side_fe->get_xyz();

  //normal vector
  const std::vector<Point> &face_normals = side_fe->get_normals();

  // The number of local degrees of freedom in each variable
  const unsigned int n_c_dofs = ctxt.get_dof_indices(c_var).size();

  // The subvectors and submatrices we need to fill:
  DenseSubMatrix<Number> &J = ctxt.get_elem_jacobian(c_var, c_var);
  DenseSubVector<Number> &R = ctxt.get_elem_residual(c_var);

  unsigned int n_qpoints = ctxt.get_side_qrule().n_points();

  bool isWest = ctxt.has_side_boundary_id(4);

  //set (in)flux boundary condition on west side
  //homogeneous neumann (Danckwerts) outflow boundary condition on east side
  //no-flux (equivalently, homoegenous neumann) boundary conditions on north, south, top, bottom sides
  //"strong" enforcement of boundary conditions
  for (unsigned int qp=0; qp != n_qpoints; qp++)
  {
    Number c = ctxt.side_value(c_var, qp);
    Gradient grad_c = ctxt.side_gradient(c_var, qp);

    //velocity vector
    NumberVectorValue U(vx, 0.0, 0.0);

    for (unsigned int i=0; i != n_c_dofs; i++)
    {
      if(isWest) //west boundary
      {
        R(i) += JxW[qp]*(U*face_normals[qp]*c - bsource*vx)*phi[i][qp];
      }

      if(request_jacobian && context.get_elem_solution_derivative())
      {
        for (unsigned int j=0; j != n_c_dofs; j++)
	      {
          if(isWest)
            J(i,j) += JxW[qp]*(U*face_normals[qp]*phi[j][qp])*phi[i][qp];
	      }
      } // end - if (request_jacobian && context.get_elem_solution_derivative())
    } //end of outer dof (i) loop
  }

  return request_jacobian;
}

//generate data
void ContamTransSys::postprocess(){
  std::ostringstream file_name;
  file_name << "Measurements" << time << ".dat";
  std::ofstream output(file_name.str());

  //location of (bottoms of) wells
  std::vector<Point> wells;
  wells.push_back(Point(497541.44, 539374.57, 8.23)); //R-1
  wells.push_back(Point(499882.61, 539296.05, 5.57)); //R-11
  wells.push_back(Point(500174.36, 538579.77, 36.73)); //R-13
  wells.push_back(Point(498442.06, 538969.46, 0)); //R-15
  wells.push_back(Point(499563.69, 538995.82, 13.15)); //R-28
  wells.push_back(Point(497855.44, 539040.32, 4.57)); //R-33#1
  wells.push_back(Point(497855.44, 539040.32, 39.62)); //R-33#2
  wells.push_back(Point(500972.02, 537650.74, 26.73)); //R-34
  wells.push_back(Point(500581.13, 539285.95, 69.01)); //R-35a
  wells.push_back(Point(500553.15, 539289.64, 11.15)); //R-35b
  wells.push_back(Point(501062.87, 538806.13, 4.86)); //R-36
  wells.push_back(Point(499174, 539122.84, 3.66)); //R-42
  wells.push_back(Point(499029.6, 539378.56, 3.32)); //R-43#1
  wells.push_back(Point(499029.6, 539378.56, 23.2)); //R-43#2
  wells.push_back(Point(499890.7, 538615.08, 4.94)); //R-44#1
  wells.push_back(Point(499890.7, 538615.08, 32.46)); //R-44#2
  wells.push_back(Point(499948.08, 538891.8, 3.59)); //R-45#1
  wells.push_back(Point(499948.08, 538891.8, 32.51)); //R-45#2
  wells.push_back(Point(499465.44, 538608.21, 3.05)); //R-50#1
  wells.push_back(Point(499465.44, 538608.21, 32.92)); //R-50#2
  wells.push_back(Point(498987.1, 538710.37, 7.62)); //R-61#1
  wells.push_back(Point(498987.1, 538710.37, 36.58)); //R-61#2
  wells.push_back(Point(498574.44, 539304.64, 4.57)); //R-62

  //write out data (get concentration at bottom of wells)
  for(int i=0; i<wells.size(); i++){
    Point pt = wells[i];
    Number c = point_value(c_var, pt);
    if(output.is_open()){
      output << pt(0) << "  " << pt(1) << "  " << pt(2) << " " << c << "\n";
    }
  }

  output.close();
}

//source term at point pt
Point ContamTransSys::forcing(const Point& pt)
{
  Point f;
/*
  if(pt(0) >= xlim[0] && pt(0) <= xlim[1] && 
     pt(1) >= ylim[0] && pt(1) <= ylim[1] && 
     pt(2) >= source_zmax-source_dz)
    f(0) = source_rate*source_conc/(water_density*source_vol); // ppb/s
  else if(pt(0) >= xlim[0] && pt(0) <= xlim[1] && 
          pt(1) >= ylim[0] && pt(1) <= ylim[1] && 
          this->get_mesh().mesh_dimension() == 2) //test if making it 2D removes oscillations
    f(0) = source_rate*source_conc/(water_density*source_vol); // ppb/s
  else
    f(0) = 0.0;
*/

  //DEBUG - try to use smoothed box source
  double dist = 0.0; //distance for decay
  if(this->get_mesh().mesh_dimension() == 3)
    dist = sqrt(pow(std::max(std::max(pt(0)-xlim[1],0.),std::max(xlim[0]-pt(0),0.)),2.) 
                    + pow(std::max(std::max(pt(1)-ylim[1],0.),std::max(ylim[0]-pt(1),0.)),2.) 
                    + pow(std::max(std::max(pt(2)-source_zmax,0.),std::max(source_zmax-source_dz-pt(2),0.)),2.)); 
  else if(this->get_mesh().mesh_dimension() == 2)
    dist = sqrt(pow(std::max(std::max(pt(0)-xlim[1],0.),std::max(xlim[0]-pt(0),0.)),2.) 
                    + pow(std::max(std::max(pt(1)-ylim[1],0.),std::max(ylim[0]-pt(1),0.)),2.)); 
  f(0) = (source_rate*source_conc/(water_density*source_vol))*exp(-ds*dist); // ppb/s

/*  
  //DEBUG - see if smoother source fixes the ridges problem...
  double xcent = 0.5*(xlim[0]+xlim[1]);
  double ycent = 0.5*(ylim[0]+ylim[1]);
  double zcent = source_zmax - 0.5*source_dz;
  f(0) = source_rate*source_conc/(water_density*source_vol)*exp(-ds*(pow(pt(0)-xcent,2.0)+pow(pt(1)-ycent,2.0)+pow(pt(2)-zcent,2.0)));
*/
  return f;
}
