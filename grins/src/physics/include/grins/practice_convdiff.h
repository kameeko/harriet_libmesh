#ifndef GRINS_PRAC_CONVDIFF_H
#define GRINS_PRAC_CONVDIFF_H

// GRINS
#include "grins/grins_enums.h"
#include "grins/physics.h"

//libMesh
#include "libmesh/enum_order.h"
#include "libmesh/enum_fe_family.h"

namespace GRINS{

	class PracticeConvDiff : public Physics{
	
	public:
	
		PracticeConvDiff( const GRINS::PhysicsName& physics_name, const GetPot& input);
		~PracticeConvDiff();
		
		//! Initialize variables for this physics.
    virtual void init_variables( libMesh::FEMSystem* system );

    //! Initialize context for added physics variables
    virtual void init_context( AssemblyContext& context );
    
    //! Time dependent part(s) of physics for element interiors
    virtual void element_time_derivative( bool compute_jacobian,
					  AssemblyContext& context,
					  CachedValues& cache );
	
	protected:
	
		libMesh::Real forcing( const libMesh::Point& p);
		
		libMesh::Number _k; //diffusion coefficient
		VariableIndex _c_var; //index for concentration
		
		libMesh::Real _source_x, _source_y; //center of source
		
		GRINSEnums::FEFamily _fefamily; //element type
	
	private:
	
		PracticeConvDiff(); //not sure what the point of this is, but the others have it...
		
	  //velocity field
		std::vector<libMesh::Real> x_pts;
		std::vector<std::vector<libMesh::Real> > y_pts;
		std::vector<std::vector<libMesh::NumberVectorValue> > vel_field;

	}; //end class

} //end namespace

#endif //GRINS_PRAC_CONVDIFF_H
