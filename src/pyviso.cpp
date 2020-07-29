/*
Copyright 2012. All rights reserved.
Institute of Measurement and Control Systems
Karlsruhe Institute of Technology, Germany

This file is part of libviso2.
Authors: Andreas Geiger

libviso2 is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or any later version.

libviso2 is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
libviso2; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA 
*/

/*
  Documented C++ sample code of stereo visual odometry (modify to your needs)
  To run this demonstration, download the Karlsruhe dataset sequence
  '2010_03_09_drive_0019' from: www.cvlibs.net!
  Usage: ./viso2 path/to/sequence/2010_03_09_drive_0019
*/

#include <iostream>
#include <string>
#include <vector>
#include <stdint.h>
#include <fstream>

#include <matcher.h>
#include <matrix.h>
#include<viso_mono.h>
#include <png++/png.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
using namespace std;

namespace py=pybind11;
// make a short name for vector of vector
using vvd = std::vector<std::vector<double>>;
using vvf = std::vector<std::vector<float>>;


class VisoMomo{
    public:
    VisoMomo(float f,float cu,float cv)
    {
      VisualOdometryMono::parameters param;
      
      // calibration parameters for sequence 2010_03_09_drive_0019 
      param.calib.f  = f; // focal length in pixels
      param.calib.cu = cu; // principal point (u-coordinate) in pixels
      param.calib.cv = cv; // principal point (v-coordinate) in pixels
      
      // init visual odometry
      viso = new VisualOdometryMono(param);
      last_move_flag = true;
    }

    bool process(string image_name)
    {
        bool move_flag = true;
        motion.clear();
        feature3d.clear();
        feature2d.clear();
        png::image< png::gray_pixel > left_img(image_name);

        // image dimensions
        int32_t width  = left_img.get_width();
        int32_t height = left_img.get_height();

        // convert input images to uint8_t buffer
        uint8_t* left_img_data  = (uint8_t*)malloc(width*height*sizeof(uint8_t));
        int32_t k=0;
        for (int32_t v=0; v<height; v++) {
            for (int32_t u=0; u<width; u++) {
              left_img_data[k]  = left_img.get_pixel(u,v);
              k++;
            }
        }

          // status
          
          // compute visual odometry
          int32_t dims[] = {width,height,width};
          std::cout<<last_move_flag<<std::endl;
          bool replace = !last_move_flag & (frame_id>1);  
          if (viso->process(left_img_data,dims,replace)) {
          
            // on success, update current pose
            for(int i=0;i<12;i++)
            {
                Matrix mo = Matrix::inv(viso->getMotion());
                mo = Matrix::reshape(mo.getMat(0,0,2,3),1,12);
                float v = mo.val[0][i];
                motion.push_back(v);
            }
        } 
        else {
            move_flag = false;
            float motion_l[12]={1,0,0,0,0,1,0,0,0,0,1,0};
            for(int i=0;i<12;i++)
            {
                motion.push_back(motion_l[i]);
            }
      }
        vector<Matcher::p_match>  matched_feature = viso->getMatches ();    
        Matrix feature3d_l = viso->getFeature3D();
        for(int i=0;i<feature3d_l.n;i++)
        {
            vector<float> f;
            for(int j=0;j<3;j++)
            {
                f.push_back(feature3d_l.val[j][i]);
            }
            feature3d.push_back(f);
        }
        for(int i=0;i<matched_feature.size();i++)
        {
            vector<float> f;
            f.push_back(matched_feature[i].u1p);
            f.push_back(matched_feature[i].v1p);
            feature2d.push_back(f);
        }


      // release uint8_t buffers
      free(left_img_data);
      last_move_flag = move_flag;
      frame_id++;
      return move_flag;  

    }
    vector<float> get_motion(){return motion;}
    vvf get_feature2d(){return feature2d;}
    vvf get_feature3d(){return feature3d;}
    private:
    VisualOdometryMono *viso;
    vector<float> motion;
    vvf feature3d;
    vvf feature2d;
    int frame_id=0;
    bool last_move_flag;
};
PYBIND11_MODULE(pyviso,m)
{   
    py::class_<VisoMomo>(m,"VisoMomo")
        .def(py::init<float,float ,float>())
        .def("process",&VisoMomo::process)
        .def("get_motion",&VisoMomo::get_motion)
        .def("get_feature2d",&VisoMomo::get_feature2d)
        .def("get_feature3d",&VisoMomo::get_feature3d);
}
