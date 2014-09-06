#include <iostream>
#include <algorithm>
#include <math.h>

#include "libmesh/libmesh.h"
#include "libmesh/mesh.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/exodusII_io.h"
#include "libmesh/equation_systems.h"
#include "libmesh/fe.h"
#include "libmesh/quadrature_gauss.h"
#include "libmesh/dof_map.h"
#include "libmesh/sparse_matrix.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/dense_matrix.h"
#include "libmesh/dense_vector.h"
#include "libmesh/linear_implicit_system.h"
#include "libmesh/perf_log.h"

#include "libmesh/dense_submatrix.h"
#include "libmesh/dense_subvector.h"

#include "libmesh/elem.h"

using namespace libMesh;

void assemble_stokesConvDiff (EquationSystems& es, const std::string& system_name);

int main (int argc, char** argv){
  LibMeshInit init (argc, argv);

  Mesh mesh(init.comm());

  MeshTools::Generation::build_square (mesh,
                                       15, 15,
                                       0., 1.,
                                       0., 1.,
                                       QUAD9);

  mesh.print_info();

  EquationSystems equation_systems (mesh);

  LinearImplicitSystem & system =
    equation_systems.add_system<LinearImplicitSystem> ("StokesConvDiff");

	//LBB-stable pressure-velocity pair
  system.add_variable ("u", SECOND);
  system.add_variable ("v", SECOND);
  system.add_variable ("p", FIRST);
  
  system.add_variable("c", FIRST);

  system.attach_assemble_function (assemble_stokesConvDiff);

  equation_systems.init ();

  equation_systems.parameters.set<unsigned int>("linear solver maximum iterations") = 250;
  equation_systems.parameters.set<Real>        ("linear solver tolerance") = 1.e-6;
  
  // The number of steps and the stopping criterion are also required
  // for the nonlinear iterations.
  const unsigned int n_nonlinear_steps = 15;
  const Real nonlinear_tolerance       = 1.e-3;

	//get reference to system
	LinearImplicitSystem & stokes_cd_system =
    equation_systems.get_system<LinearImplicitSystem> ("StokesConvDiff");
	
	//get copy of solution at current nonlinear iteration
	AutoPtr<NumericVector<Number> >
		last_nonlinear_soln(stokes_cd_system.solution->clone()); //doesn't quite match ex2

  equation_systems.print_info();
  
  // Create a performance-logging object for this example
  PerfLog perf_log("Steady Coupled Stokes-Convection-Diffusion");

	for(unsigned int l=0; l<n_nonlinear_steps; ++l){
		//update nonlinear solution
		last_nonlinear_soln->zero();
		last_nonlinear_soln->add(*stokes_cd_system.solution);
		
		//assemble and solve linear system
		perf_log.push("linear solve");
		equation_systems.get_system("StokesConvDiff").solve();
		perf_log.pop("linear solve");
		
		//computer difference from last nonlinear iterate
		last_nonlinear_soln->add(-1.0, *stokes_cd_system.solution);
		
		//close vector before computing its norm
		last_nonlinear_soln->close();
		
		//compute l2 norm of difference
		const Real norm_delta = last_nonlinear_soln->l2_norm();
    if(isnan(norm_delta))
    	std::cout << "DON'T FORGET TO USE --use-laspack\n";
		
		//number of iterations required to solve linear system
		const unsigned int n_linear_iterations = stokes_cd_system.n_linear_iterations();
		
		//final residual of linear system
		const Real final_linear_residual = stokes_cd_system.final_linear_residual();
		
    // Print out convergence information for the linear and nonlinear iterations.    	
    std::cout << "Linear solver converged at step: "
              << n_linear_iterations
              << ", final residual: "
              << final_linear_residual
              << "  Nonlinear convergence: ||u - u_old|| = "
              << norm_delta
              << std::endl;

    // Terminate the solution iteration if the difference between
    // this nonlinear iterate and the last is sufficiently small, AND
    // if the most recent linear system was solved to a sufficient tolerance.
    if ((norm_delta < nonlinear_tolerance) &&
        (stokes_cd_system.final_linear_residual() < nonlinear_tolerance)){
        
      std::cout << " Nonlinear solver converged at step "
                << l
                << std::endl;
      break;
    }
	} //end nonlinear solve
	  
#ifdef LIBMESH_HAVE_EXODUS_API
  ExodusII_IO(mesh).write_equation_systems ("out.exo", equation_systems);
#endif // #ifdef LIBMESH_HAVE_EXODUS_API

  return 0;
} //end main

void assemble_stokesConvDiff (EquationSystems& es, const std::string& system_name){
  libmesh_assert_equal_to (system_name, "StokesConvDiff");

  const MeshBase& mesh = es.get_mesh();

  const unsigned int dim = mesh.mesh_dimension();

  LinearImplicitSystem & system =
    es.get_system<LinearImplicitSystem> ("StokesConvDiff");

  const unsigned int u_var = system.variable_number ("u");
  const unsigned int v_var = system.variable_number ("v");
  const unsigned int p_var = system.variable_number ("p");
  const unsigned int c_var = system.variable_number("c");

  FEType fe_vel_type = system.variable_type(u_var);
  FEType fe_pres_type = system.variable_type(p_var);
  FEType fe_conc_type = system.variable_type(c_var);

  AutoPtr<FEBase> fe_vel  (FEBase::build(dim, fe_vel_type));
  AutoPtr<FEBase> fe_pres (FEBase::build(dim, fe_pres_type));
  AutoPtr<FEBase> fe_conc(FEBase::build(dim, fe_conc_type));

  QGauss qrule (dim, fe_vel_type.default_quadrature_order());

  fe_vel->attach_quadrature_rule (&qrule);
  fe_pres->attach_quadrature_rule (&qrule);
  fe_conc->attach_quadrature_rule(&qrule);

  const std::vector<Real>& JxW = fe_vel->get_JxW(); //quadrature weights

	const std::vector<Point>& q_point = fe_conc->get_xyz(); //locations of quadrature points

	//velocities have same shape function
	const std::vector<std::vector<Real> >& phi = fe_vel->get_phi();
  const std::vector<std::vector<RealGradient> >& dphi = fe_vel->get_dphi();

	//pressure and concentration have same shape functions in this case
  const std::vector<std::vector<Real> >& psi = fe_pres->get_phi(); 
  const std::vector<std::vector<RealGradient> >& dpsi = fe_conc->get_dphi();

  const DofMap & dof_map = system.get_dof_map();

  DenseMatrix<Number> Ke;
  DenseVector<Number> Fe;

  DenseSubMatrix<Number>
    Kuu(Ke), Kuv(Ke), Kup(Ke), Kuc(Ke),
    Kvu(Ke), Kvv(Ke), Kvp(Ke), Kvc(Ke),
    Kpu(Ke), Kpv(Ke), Kpp(Ke), Kpc(Ke),
    Kcu(Ke), Kcv(Ke), Kcp(Ke), Kcc(Ke);

  DenseSubVector<Number>
    Fu(Fe),
    Fv(Fe),
    Fp(Fe),
    Fc(Fe);

  std::vector<dof_id_type> dof_indices;
  std::vector<dof_id_type> dof_indices_u;
  std::vector<dof_id_type> dof_indices_v;
  std::vector<dof_id_type> dof_indices_p;
	std::vector<dof_id_type> dof_indices_c;

  MeshBase::const_element_iterator       el     = mesh.active_local_elements_begin();
  const MeshBase::const_element_iterator end_el = mesh.active_local_elements_end();

  for ( ; el != end_el; ++el)
    {
      const Elem* elem = *el;

      dof_map.dof_indices (elem, dof_indices);
      dof_map.dof_indices (elem, dof_indices_u, u_var);
      dof_map.dof_indices (elem, dof_indices_v, v_var);
      dof_map.dof_indices (elem, dof_indices_p, p_var);
      dof_map.dof_indices(elem, dof_indices_c, c_var);

      const unsigned int n_dofs   = dof_indices.size();
      const unsigned int n_u_dofs = dof_indices_u.size();
      const unsigned int n_v_dofs = dof_indices_v.size();
      const unsigned int n_p_dofs = dof_indices_p.size();
      const unsigned int n_c_dofs = dof_indices_c.size();

      fe_vel->reinit  (elem);
      fe_pres->reinit (elem);
      fe_conc->reinit(elem);

      Ke.resize (n_dofs, n_dofs);
      Fe.resize (n_dofs);

      Kuu.reposition (u_var*n_u_dofs, u_var*n_u_dofs, n_u_dofs, n_u_dofs);
      Kuv.reposition (u_var*n_u_dofs, v_var*n_u_dofs, n_u_dofs, n_v_dofs);
      Kup.reposition (u_var*n_u_dofs, p_var*n_u_dofs, n_u_dofs, n_p_dofs);
      Kuc.reposition(u_var*n_u_dofs, 2*n_u_dofs + n_p_dofs, n_u_dofs, n_c_dofs);

      Kvu.reposition (v_var*n_v_dofs, u_var*n_v_dofs, n_v_dofs, n_u_dofs);
      Kvv.reposition (v_var*n_v_dofs, v_var*n_v_dofs, n_v_dofs, n_v_dofs);
      Kvp.reposition (v_var*n_v_dofs, p_var*n_v_dofs, n_v_dofs, n_p_dofs);
      Kvc.reposition(v_var*n_v_dofs, 2*n_u_dofs + n_p_dofs, n_v_dofs, n_c_dofs);

      Kpu.reposition (p_var*n_u_dofs, u_var*n_u_dofs, n_p_dofs, n_u_dofs);
      Kpv.reposition (p_var*n_u_dofs, v_var*n_u_dofs, n_p_dofs, n_v_dofs);
      Kpp.reposition (p_var*n_u_dofs, p_var*n_u_dofs, n_p_dofs, n_p_dofs);
      Kpc.reposition(p_var*n_u_dofs, 2*n_u_dofs + n_p_dofs, n_p_dofs, n_c_dofs);
      
      Kcu.reposition(2*n_u_dofs + n_p_dofs, 0, n_c_dofs, n_u_dofs);
      Kcv.reposition(2*n_u_dofs + n_p_dofs, n_u_dofs, n_c_dofs, n_v_dofs);
      Kcp.reposition(2*n_u_dofs + n_p_dofs, 2*n_u_dofs, n_c_dofs, n_p_dofs);
      Kcc.reposition(2*n_u_dofs + n_p_dofs, 2*n_u_dofs + n_p_dofs, n_c_dofs, n_c_dofs);

      Fu.reposition (u_var*n_u_dofs, n_u_dofs);
      Fv.reposition (v_var*n_u_dofs, n_v_dofs);
      Fp.reposition (p_var*n_u_dofs, n_p_dofs);
      Fc.reposition(2*n_u_dofs + n_p_dofs, n_c_dofs);

      for (unsigned int qp=0; qp<qrule.n_points(); qp++){
		    //solution at previous nonlinear iteration
		    Number uprevnl = 0.0; 
		    Number vprevnl = 0.0;
		    for(unsigned int l=0; l<n_u_dofs; l++){
		    	uprevnl += phi[l][qp]*system.current_solution(dof_indices_u[l]);
		    	vprevnl += phi[l][qp]*system.current_solution(dof_indices_v[l]);
		    }
		    
  	  	//location of quadrature point
		    const Real x = q_point[qp](0);
		    const Real y = q_point[qp](1);
		    
      	//pressure-velocity relation
        for (unsigned int i=0; i<n_u_dofs; i++)
          for (unsigned int j=0; j<n_u_dofs; j++)
            Kuu(i,j) += JxW[qp]*(dphi[i][qp]*dphi[j][qp]);

        for (unsigned int i=0; i<n_u_dofs; i++)
          for (unsigned int j=0; j<n_p_dofs; j++)
            Kup(i,j) += -JxW[qp]*psi[j][qp]*dphi[i][qp](0);

				//pressure-velocity relation
        for (unsigned int i=0; i<n_v_dofs; i++)
          for (unsigned int j=0; j<n_v_dofs; j++)
            Kvv(i,j) += JxW[qp]*(dphi[i][qp]*dphi[j][qp]);

        for (unsigned int i=0; i<n_v_dofs; i++)
          for (unsigned int j=0; j<n_p_dofs; j++)
            Kvp(i,j) += -JxW[qp]*psi[j][qp]*dphi[i][qp](1);

				//zero-divergence velocity
        for (unsigned int i=0; i<n_p_dofs; i++)
          for (unsigned int j=0; j<n_u_dofs; j++)
            Kpu(i,j) += -JxW[qp]*psi[i][qp]*dphi[j][qp](0);

        for (unsigned int i=0; i<n_p_dofs; i++)
          for (unsigned int j=0; j<n_v_dofs; j++)
            Kpv(i,j) += -JxW[qp]*psi[i][qp]*dphi[j][qp](1);
            
        //convection-diffusion
        for(unsigned int i=0; i<n_c_dofs; i++)
        	for(unsigned int j=0; j<n_c_dofs; j++)
        		Kcc(i,j) += JxW[qp]*(dpsi[i][qp]*dpsi[j][qp]) //diffusion term
        			-JxW[qp]*(uprevnl*dpsi[i][qp](0)*psi[j][qp] + vprevnl*dpsi[i][qp](1)*psi[j][qp]); //convection term
        			//with wonky sign in convection term, for steady_convdiff weirdness...

				{ //scope bubble
		    const Real fxy = exp(-10*(pow(x-0.25,2)+pow(y-0.25,2))); //forcing function
		    for (unsigned int i=0; i<psi.size(); i++)
		      Fc(i) += JxW[qp]*fxy*psi[i][qp];
		    } //end scope bubble
      } // end of the quadrature point qp-loop

      { //boundary conditions - lid-cavity for velocity, neumann for concentration...
        for (unsigned int s=0; s<elem->n_sides(); s++)
          if (elem->neighbor(s) == NULL)
            {
              AutoPtr<Elem> side (elem->build_side(s));

              for (unsigned int ns=0; ns<side->n_nodes(); ns++)
                {

                  const Real yf = side->point(ns)(1);

                  const Real penalty = 1.e10;


                  const Real u_value = (yf > .99) ? 1. : 0.;

                  const Real v_value = 0.;

                  for (unsigned int n=0; n<elem->n_nodes(); n++)
                    if (elem->node(n) == side->node(ns))
                      {
                        Kuu(n,n) += penalty;
                        Kvv(n,n) += penalty;

                        Fu(n) += penalty*u_value;
                        Fv(n) += penalty*v_value;
                      }
                } // end face node loop
            } // end if (elem->neighbor(side) == NULL)
      } // end boundary condition section

      dof_map.constrain_element_matrix_and_vector (Ke, Fe, dof_indices);

      system.matrix->add_matrix (Ke, dof_indices);
      system.rhs->add_vector    (Fe, dof_indices);
    } // end of element loop

  return;
}
