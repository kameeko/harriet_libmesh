#include "grins/practice_bc_handling.h"

// libMesh
#include "libmesh/const_function.h"
#include "libmesh/dirichlet_boundaries.h"
#include "libmesh/dof_map.h"
#include "libmesh/fem_system.h"

namespace GRINS
{

	PracticeBCHandling::PracticeBCHandling(const std::string& physics_name,
						 const GetPot& input)
    : BCHandlingBase(physics_name){
    
    std::string id_str = "Physics/"+_physics_name+"/bc_ids";
    std::string bc_str = "Physics/"+_physics_name+"/bc_types";
    std::string var_str = "Physics/"+_physics_name+"/bc_variables";
    std::string val_str = "Physics/"+_physics_name+"/bc_values";
    
    this->read_bc_data( input, id_str, bc_str, var_str, val_str );
    
    return;
  }

	PracticeBCHandling::~PracticeBCHandling(){return;}
	
	void PracticeBCHandling::init_bc_data( const libMesh::FEMSystem& system ){
		libmesh_assert( system.has_variable("c"));
		_c_var = system.variable_number("c");
		if(system.has_variable("zc")){
			_zc_var = system.variable_number("zc");
			_has_zc = true;
		}
		else
			_has_zc = false;
			
		if(system.has_variable("fc")){
			_fc_var = system.variable_number("fc");
			_has_fc = true;
		}
		else
			_has_fc = false;
			
		if(system.has_variable("auxc")){
			_aux_c_var = system.variable_number("auxc");
			_has_auxc = true;
		}
		else
			_has_auxc = false;
			
		if(system.has_variable("auxzc")){
			_aux_zc_var = system.variable_number("auxzc");
			_has_auxzc = true;
		}
		else
			_has_auxzc = false;
			
		if(system.has_variable("auxfc")){
			_aux_fc_var = system.variable_number("auxfc");
			_has_auxfc = true;
		}
		else
			_has_auxfc = false;
			
		return;
	}
	
	void PracticeBCHandling::init_bc_types(const BoundaryID bc_id, 
					      const std::string& bc_id_string, 
					      const int bc_type, 
					      const std::string& bc_vars, 
					      const std::string& bc_value, 
					      const GetPot& input ){
					      
    this->set_dirichlet_bc_type( bc_id, bc_type );
	  this->set_dirichlet_bc_value( bc_id, 0.0 ); //true for all variables in practice cases
				      
	}
	
	void PracticeBCHandling::user_init_dirichlet_bcs( libMesh::FEMSystem* /*system*/,
							libMesh::DofMap& dof_map,
							BoundaryID bc_id,
							BCType bc_type ) const{
	
		std::set<BoundaryID> dbc_ids;
	  dbc_ids.insert(bc_id);

	  std::vector<VariableIndex> dbc_vars;
	  dbc_vars.push_back(_c_var);
	  if(_has_zc)
	  	dbc_vars.push_back(_zc_var);
	  if(_has_fc)
	  	dbc_vars.push_back(_fc_var);
	  if(_has_auxc)
	  	dbc_vars.push_back(_aux_c_var);
	  if(_has_auxzc)
	  	dbc_vars.push_back(_aux_zc_var);
	  if(_has_auxfc)
	  	dbc_vars.push_back(_aux_fc_var);
	
    libMesh::ConstFunction<libMesh::Number> c_func(this->get_dirichlet_bc_value(bc_id));
	
	  libMesh::DirichletBoundary c_dbc( dbc_ids, dbc_vars, &c_func );
	
	  dof_map.add_dirichlet_boundary( c_dbc );	
	
	}

} //namespace GRINS
