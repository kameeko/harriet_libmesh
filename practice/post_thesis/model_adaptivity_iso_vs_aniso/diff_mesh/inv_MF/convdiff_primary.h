#include "libmesh/fem_system.h"
#include "libmesh/getpot.h"
#include "libmesh/elem.h"
#include "libmesh/point_locator_tree.h"

using namespace libMesh;

class ConvDiff_PrimarySys : public FEMSystem
{
public:

  // Constructor
  ConvDiff_PrimarySys(EquationSystems& es,
               const std::string& name_in,
               const unsigned int number_in)
    : FEMSystem(es, name_in, number_in){
    
    GetPot infile("contamTrans.in");
		std::string find_data_here = infile("data_file","Measurements0.dat");
		qoi_option = infile("QoI_option",1);
    solveInit = (!(infile("solveMF",true)) && !infile("solveHF",false));
    const unsigned int dim = this->get_mesh().mesh_dimension();
    
    //read in data
		if(FILE *fp=fopen(find_data_here.c_str(),"r")){
			Real x, y, z, value;
			int flag = 1;
			while(flag != -1){
				flag = fscanf(fp,"%lf %lf %lf %lf",&x,&y,&z,&value);
				if(flag != -1){
          if(dim == 3)
					  datapts.push_back(Point(x,y,z));
          else if(dim == 2)
            datapts.push_back(Point(x,y,0.));
					datavals.push_back(value);
				}
			}
			fclose(fp);
	  }else{
	    std::cout << "\n\nAAAAAHHHHH NO DATA FOUND?!?!\n\n" << std::endl;
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
                                                                              
  //to calculate QoI
  virtual void element_postprocess(DiffContext &context);

  // Indices for each variable;
  unsigned int c_var, zc_var, fc_var;
  
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
	
	int cd_subdomain_id, cdr_subdomain_id;
  bool solveInit;
	
  //options for QoI location and nature
  int qoi_option;
  Real qoi;
  double getQoI(){ return qoi; }
  void clearQoI(){ qoi = 0.; }

  //update elements where data points reside when mesh is changed
  void updateDataLoc(){
    PointLocatorTree point_locator(this->get_mesh());
    for(unsigned int ind=0; ind<dataelems.size(); ind++){
      if(!(this->get_mesh().elem(dataelems[ind])->active())){ //element has been refined
        Point data_point = datapts[ind];
        Elem *this_elem = const_cast<Elem *>(point_locator(data_point));
        dataelems[ind] = this_elem->id();
      }
    }
  }
 
  virtual void postprocess();
  std::vector<Real> primal_c_vals;
  std::vector<Real> get_c_vals(){ return primal_c_vals; }
  
};