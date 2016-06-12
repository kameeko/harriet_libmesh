#include "libmesh/fem_system.h"
#include "libmesh/getpot.h"
#include "libmesh/point_locator_tree.h"

using namespace libMesh;

class ConvDiff_AuxSadjSys : public FEMSystem
{
public:

  // Constructor
  ConvDiff_AuxSadjSys(EquationSystems& es,
               const std::string& name_in,
               const unsigned int number_in)
    : FEMSystem(es, name_in, number_in){
    
    GetPot infile("contamTrans.in");
		std::string find_data_here = infile("data_file","Measurements0.dat");
		qoi_option = infile("QoI_option",1);
    
    //read in data
		if(FILE *fp=fopen(find_data_here.c_str(),"r")){
			Real x, y, z, value;
			int flag = 1;
			while(flag != -1){
				flag = fscanf(fp,"%lf %lf %lf %lf",&x,&y,&z,&value);
				if(flag != -1){
					datapts.push_back(Point(x,y,z));
					datavals.push_back(value);
				}
			}
			fclose(fp);
	  }
	  
	  //read in auxiliary variables at data points
	  if(FILE *fp=fopen("auxc_points.dat","r")){
	    int flag = 1;
	    Real meep;
	    while(flag != -1){
	      flag = fscanf(fp,"%lf",&meep);
	      if(flag != -1)
	        primal_auxc_vals.push_back(meep);
	    }
	    fclose(fp);
	  }
	  
	  //find elements in which data points reside
	  PointLocatorTree point_locator(this->get_mesh());
	  for(unsigned int dnum=0; dnum<datavals.size(); dnum++){
	  	Point data_point = datapts[dnum];
	  	Elem *this_elem = const_cast<Elem *>(point_locator(data_point));
	  	dataelems.push_back(this_elem->id());
	  }
	  
  }

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

  // Indices for each variable;
  unsigned int aux_c_var, aux_zc_var, aux_fc_var;
  
  Real beta; //regularization parameter
  Real vx; //west-east velocity (along x-axis, for now); m/s
  Real react_rate; //reaction rate; 1/s
  Real porosity; //porosity
  Real bsource; // ppb (concentration of influx at west boundary)
  NumberTensorValue dispTens; //dispersion tensor; m^2/s
  
  //data-related stuff
  std::vector<Point> datapts; 
  std::vector<Real> datavals;
  std::vector<dof_id_type> dataelems;
  
  //options for QoI location and nature
  int qoi_option;
  
  std::vector<Real> primal_auxc_vals;
  
};
